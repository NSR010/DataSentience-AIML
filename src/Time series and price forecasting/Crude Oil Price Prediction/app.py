import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import joblib

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_data(path):
    df = pd.read_csv(path)
    date_col = None
    price_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            date_col = c
        if any(k in c.lower() for k in ['price','close','adj close','oil','spot']):
            price_col = c
    if date_col is None:
        date_col = df.columns[0]
    if price_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols)>0:
            price_col = num_cols[0]
        else:
            raise ValueError("No suitable price column found in the dataset. Please ensure the CSV has a numeric column for prices.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    ts = df[[date_col, price_col]].rename(columns={date_col:'Date', price_col:'Price'}).set_index('Date')
    ts = ts.asfreq('D')
    ts['Price'] = ts['Price'].interpolate(method='time').ffill().bfill()
    return ts

def plot_and_save(ts, col_name, title, filename):
    plt.figure(figsize=(12,4))
    plt.plot(ts[col_name])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    st.pyplot(plt)
    plt.close()

def create_lag_features(series, lags=5):
    df = pd.DataFrame({'Date':series.index, 'Price':series.values})
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['Price'].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df

def run_random_forest(ts, test_size_ratio=0.2, lags=5):
    test_size = int(len(ts) * test_size_ratio)
    train = ts.iloc[:-test_size].copy()
    test = ts.iloc[-test_size:].copy()
    rf_df = create_lag_features(ts['Price'], lags=lags)
    total = len(ts)
    train_size = total - test_size
    rf_train = rf_df.iloc[:max(0, train_size-lags)].reset_index(drop=True)
    rf_test = rf_df.iloc[max(0, train_size-lags): max(0, train_size-lags) + test_size].reset_index(drop=True)
    X_train = rf_train[[f'lag_{i}' for i in range(1,lags+1)]].values
    y_train = rf_train['Price'].values
    X_test = rf_test[[f'lag_{i}' for i in range(1,lags+1)]].values
    y_test = rf_test['Price'].values
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = root_mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mape = mape(y_test, rf_pred)
    return rf_test['Date'], y_test, rf_pred, rf_rmse, rf_mae, rf_mape

def run_arima(ts, test_size_ratio=0.2, order=(5,1,0)):
    test_size = int(len(ts) * test_size_ratio)
    train = ts.iloc[:-test_size].copy()
    test = ts.iloc[-test_size:].copy()
    try:
        arima_model = ARIMA(train['Price'], order=order)
        arima_res = arima_model.fit()
        arima_forecast_result = arima_res.get_forecast(steps=len(test))
        arima_pred = arima_forecast_result.predicted_mean.values
        arima_rmse = root_mean_squared_error(test['Price'], arima_pred)
        arima_mae = mean_absolute_error(test['Price'], arima_pred)
        arima_mape = mape(test['Price'].values, arima_pred)
        return test.index, test['Price'], arima_pred, arima_rmse, arima_mae, arima_mape, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)

def forecast_arima(ts, steps=30, order=(5,1,0)):
    try:
        model = ARIMA(ts['Price'], order=order)
        res = model.fit()
        forecast_result = res.get_forecast(steps=steps)
        preds = forecast_result.predicted_mean
        future_dates = preds.index
        return future_dates, preds
    except Exception as e:
        return None, str(e)

def main():
    st.sidebar.title("⚙️ Settings")
    test_ratio = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
    lags = st.sidebar.slider("Lag Features for RF", 3, 10, 5)
    forecast_steps = st.sidebar.slider("Forecast Days", 10, 90, 30)

    st.title("🛢️ Crude Oil Price Forecasting Dashboard")
    st.markdown("Welcome to the interactive dashboard for analyzing and forecasting crude oil prices! 📈")
    st.markdown("Explore trends, run models, and predict future prices with ease.")

    data_path = "Crude_Oil_Data.csv"
    if not os.path.exists(data_path):
        st.error(f"❌ {data_path} not found. Please place the dataset in the root folder.")
        return

    ts = load_data(data_path)
    if len(ts) == 0:
        st.error("❌ No valid data found in the CSV file. Please ensure it has date and price columns.")
        return
    st.success("✅ Data loaded successfully!")

    st.header("📊 Exploratory Data Analysis (EDA)")
    with st.expander("View EDA Details"):
        st.write("Analyze the historical price trends, rolling averages, and volatility.")

    # Price Trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts['Price'], mode='lines', name='Price', line=dict(color='blue')))
    fig.update_layout(title='Crude Oil Price Trend', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
    st.plotly_chart(fig)

    # Rolling Mean
    rolling_mean = ts['Price'].rolling(30).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ts.index, y=rolling_mean, mode='lines', name='30-Day Rolling Mean', line=dict(color='green')))
    fig2.update_layout(title='30-Day Rolling Mean', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
    st.plotly_chart(fig2)

    # Rolling Std
    rolling_std = ts['Price'].rolling(30).std()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ts.index, y=rolling_std, mode='lines', name='30-Day Rolling Std', line=dict(color='red')))
    fig3.update_layout(title='30-Day Rolling Std Dev', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
    st.plotly_chart(fig3)

    st.header("🔍 Stationarity Test")
    adf_res = adfuller(ts['Price'].dropna())
    col1, col2 = st.columns(2)
    col1.metric("ADF Statistic", f"{adf_res[0]:.4f}")
    col2.metric("P-value", f"{adf_res[1]:.4f}")
    if adf_res[1] < 0.05:
        st.success("✅ The time series is likely stationary.")
    else:
        st.warning("⚠️ The time series is likely non-stationary.")

    st.header("🌲 Random Forest Model")
    if st.button("🚀 Run Random Forest Model"):
        dates, y_true, y_pred, rmse, mae, mape_val = run_random_forest(ts, test_ratio, lags)
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("MAPE", f"{mape_val:.2f}%")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='RF Prediction', line=dict(color='orange')))
        fig.update_layout(title='Random Forest Predictions vs Actual', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
        st.plotly_chart(fig)

    st.header("📈 ARIMA Model")
    if st.button("🚀 Run ARIMA Model"):
        dates, y_true, y_pred, rmse, mae, mape_val, error = run_arima(ts, test_ratio)
        if error:
            st.error(f"❌ ARIMA model failed: {error}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("MAPE", f"{mape_val:.2f}%")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='ARIMA Prediction', line=dict(color='purple')))
            fig.update_layout(title='ARIMA Predictions vs Actual', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
            st.plotly_chart(fig)

    st.header("🔮 Forecast Future Prices")
    if st.button("🔮 Generate Forecast"):
        dates, preds = forecast_arima(ts, forecast_steps)
        if dates is None:
            st.error(f"❌ Forecast failed: {preds}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts['Price'], mode='lines', name='Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
            fig.update_layout(title=f'{forecast_steps}-Day Forecast', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white')
            st.plotly_chart(fig)

    with st.expander("ℹ️ Model Explanations"):
        st.markdown("""
        - **Random Forest**: Uses lag features to predict prices based on historical data.
        - **ARIMA**: A statistical model for time series forecasting.
        - **Forecast**: Predicts future prices using ARIMA on the full dataset.
        """)

    st.markdown("---")
    st.markdown("**Note:** LSTM model is not included due to TensorFlow complexity. Run separately if needed. 🧠")

if __name__ == "__main__":
    main()
