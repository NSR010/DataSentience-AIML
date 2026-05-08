import os
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))

    return np.mean(actual_direction == predicted_direction)


def load_data(path):
    df = pd.read_csv(path)

    date_col = None
    price_col = None

    for c in df.columns:
        col = c.lower()

        if "date" in col or "time" in col:
            date_col = c

        if any(k in col for k in ["adj_close", "adj close", "price", "close", "oil", "spot"]):
            price_col = c

    if date_col is None:
        date_col = df.columns[0]

    if price_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            price_col = num_cols[0]
        else:
            raise ValueError("No suitable price column found in the dataset.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

    ts = df[[date_col, price_col]].rename(
        columns={date_col: "Date", price_col: "Price"}
    ).set_index("Date")

    ts["Price"] = ts["Price"].interpolate().ffill().bfill()

    return ts


def evaluate_model(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Directional Accuracy": directional_accuracy(y_true, y_pred),
    }


def create_lag_features(series, lags=5):
    df = pd.DataFrame({
        "Date": series.index,
        "Price": series.values
    })

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Price"].shift(i)

    df["rolling_mean_7"] = df["Price"].rolling(7).mean()
    df["rolling_std_7"] = df["Price"].rolling(7).std()
    df["return_1d"] = df["Price"].pct_change(1)

    df = df.dropna().reset_index(drop=True)

    return df


def run_baselines(ts, test_size_ratio=0.2):
    test_size = int(len(ts) * test_size_ratio)
    test = ts.iloc[-test_size:].copy()

    y_true = test["Price"].values
    naive_pred = ts["Price"].shift(1).iloc[-test_size:].values
    ma7_pred = ts["Price"].rolling(7).mean().iloc[-test_size:].values

    return {
        "dates": test.index,
        "y_true": y_true,
        "naive_pred": naive_pred,
        "ma7_pred": ma7_pred,
        "metrics": [
            evaluate_model("Naive Baseline", y_true, naive_pred),
            evaluate_model("Moving Average 7", y_true, ma7_pred)
        ]
    }


def run_random_forest(ts, test_size_ratio=0.2, lags=5):
    test_size = int(len(ts) * test_size_ratio)

    rf_df = create_lag_features(ts["Price"], lags=lags)

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)] + [
        "rolling_mean_7",
        "rolling_std_7",
        "return_1d"
    ]

    split = len(rf_df) - test_size

    X_train = rf_df[feature_cols].iloc[:split].values
    y_train = rf_df["Price"].iloc[:split].values

    X_test = rf_df[feature_cols].iloc[split:].values
    y_test = rf_df["Price"].iloc[split:].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    return {
        "dates": rf_df["Date"].iloc[split:],
        "y_true": y_test,
        "y_pred": pred,
        "metrics": evaluate_model("Random Forest", y_test, pred)
    }


def run_xgboost(ts, test_size_ratio=0.2, lags=5):
    if not XGBOOST_AVAILABLE:
        return None, "XGBoost is not installed."

    test_size = int(len(ts) * test_size_ratio)

    xgb_df = create_lag_features(ts["Price"], lags=lags)

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)] + [
        "rolling_mean_7",
        "rolling_std_7",
        "return_1d"
    ]

    split = len(xgb_df) - test_size

    X_train = xgb_df[feature_cols].iloc[:split].values
    y_train = xgb_df["Price"].iloc[:split].values

    X_test = xgb_df[feature_cols].iloc[split:].values
    y_test = xgb_df["Price"].iloc[split:].values

    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    pred = xgb.predict(X_test)

    return {
        "dates": xgb_df["Date"].iloc[split:],
        "y_true": y_test,
        "y_pred": pred,
        "metrics": evaluate_model("XGBoost", y_test, pred)
    }, None


def run_arima(ts, test_size_ratio=0.2, order=(5, 1, 0)):
    test_size = int(len(ts) * test_size_ratio)

    train = ts.iloc[:-test_size].copy()
    test = ts.iloc[-test_size:].copy()

    try:
        model = ARIMA(train["Price"], order=order)
        result = model.fit()

        forecast = result.forecast(steps=len(test))
        pred = np.array(forecast)

        return {
            "dates": test.index,
            "y_true": test["Price"].values,
            "y_pred": pred,
            "metrics": evaluate_model("ARIMA", test["Price"].values, pred)
        }, None

    except Exception as e:
        return None, str(e)


def forecast_arima(ts, steps=30, order=(5, 1, 0)):
    try:
        model = ARIMA(ts["Price"], order=order)
        result = model.fit()

        pred = result.forecast(steps=steps)

        last_date = ts.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps
        )

        return future_dates, pred.values, None

    except Exception as e:
        return None, None, str(e)


def plot_prediction(title, dates, y_true, y_pred, pred_label):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode="lines",
        name=pred_label
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_metrics(metrics):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col2.metric("MAE", f"{metrics['MAE']:.2f}")
    col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    col4.metric("Directional Accuracy", f"{metrics['Directional Accuracy']:.2f}")


def main():
    st.set_page_config(page_title="Crude Oil Forecasting Dashboard", layout="wide")

    st.sidebar.title("⚙️ Settings")
    test_ratio = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
    lags = st.sidebar.slider("Lag Features for ML Models", 3, 20, 5)
    forecast_steps = st.sidebar.slider("Forecast Days", 10, 90, 30)

    st.title("🛢️ Crude Oil Price Forecasting Dashboard")
    st.markdown(
        "Compare baseline, statistical, and machine learning models for crude oil price forecasting."
    )

    data_path = "Crude_Oil_Data.csv"

    if not os.path.exists(data_path):
        st.error(f"❌ {data_path} not found. Please place the dataset in the root folder.")
        return

    ts = load_data(data_path)

    if len(ts) == 0:
        st.error("❌ No valid data found in the CSV file.")
        return

    st.success("✅ Data loaded successfully!")

    st.header("📊 Exploratory Data Analysis")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts["Price"],
        mode="lines",
        name="Price"
    ))

    fig.update_layout(
        title="Crude Oil Price Trend",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    rolling_mean = ts["Price"].rolling(30).mean()
    rolling_std = ts["Price"].rolling(30).std()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ts.index,
        y=rolling_mean,
        mode="lines",
        name="30-Day Rolling Mean"
    ))

    fig2.update_layout(
        title="30-Day Rolling Mean",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=ts.index,
        y=rolling_std,
        mode="lines",
        name="30-Day Rolling Std"
    ))

    fig3.update_layout(
        title="30-Day Rolling Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility",
        template="plotly_white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.header(" Stationarity Test")

    adf_res = adfuller(ts["Price"].dropna())

    col1, col2 = st.columns(2)
    col1.metric("ADF Statistic", f"{adf_res[0]:.4f}")
    col2.metric("P-value", f"{adf_res[1]:.4f}")

    if adf_res[1] < 0.05:
        st.success(" The time series is likely stationary.")
    else:
        st.warning(" The time series is likely non-stationary.")

    st.header(" Baseline Models")

    if st.button(" Run Baseline Models"):
        baseline = run_baselines(ts, test_ratio)

        results_df = pd.DataFrame(baseline["metrics"])
        st.dataframe(results_df, use_container_width=True)

        plot_prediction(
            "Naive Baseline vs Actual",
            baseline["dates"],
            baseline["y_true"],
            baseline["naive_pred"],
            "Naive Baseline"
        )

        plot_prediction(
            "7-Day Moving Average vs Actual",
            baseline["dates"],
            baseline["y_true"],
            baseline["ma7_pred"],
            "Moving Average 7"
        )

    st.header("🌲 Random Forest Model")

    if st.button("🚀 Run Random Forest Model"):
        output = run_random_forest(ts, test_ratio, lags)

        show_metrics(output["metrics"])

        plot_prediction(
            "Random Forest Predictions vs Actual",
            output["dates"],
            output["y_true"],
            output["y_pred"],
            "Random Forest"
        )

    st.header("⚡ XGBoost Model")

    if st.button("🚀 Run XGBoost Model"):
        output, error = run_xgboost(ts, test_ratio, lags)

        if error:
            st.error(f" {error}")
        else:
            show_metrics(output["metrics"])

            plot_prediction(
                "XGBoost Predictions vs Actual",
                output["dates"],
                output["y_true"],
                output["y_pred"],
                "XGBoost"
            )

    st.header("📈 ARIMA Model")

    if st.button("🚀 Run ARIMA Model"):
        output, error = run_arima(ts, test_ratio)

        if error:
            st.error(f" ARIMA model failed: {error}")
        else:
            show_metrics(output["metrics"])

            plot_prediction(
                "ARIMA Predictions vs Actual",
                output["dates"],
                output["y_true"],
                output["y_pred"],
                "ARIMA"
            )

    st.header(" Run All Models and Compare")

    if st.button(" Run Full Model Comparison"):
        all_results = []

        baseline = run_baselines(ts, test_ratio)
        all_results.extend(baseline["metrics"])

        rf_output = run_random_forest(ts, test_ratio, lags)
        all_results.append(rf_output["metrics"])

        xgb_output, xgb_error = run_xgboost(ts, test_ratio, lags)
        if xgb_error is None:
            all_results.append(xgb_output["metrics"])

        arima_output, arima_error = run_arima(ts, test_ratio)
        if arima_error is None:
            all_results.append(arima_output["metrics"])

        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("MAE").reset_index(drop=True)

        st.subheader("Model Comparison Table")
        st.dataframe(results_df, use_container_width=True)

        best_model = results_df.iloc[0]

        st.success(
            f"Best model by MAE: {best_model['Model']} "
            f"with MAE = {best_model['MAE']:.4f}"
        )

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=results_df["Model"],
            y=results_df["MAE"],
            name="MAE"
        ))

        fig_comp.update_layout(
            title="Model Comparison by MAE",
            xaxis_title="Model",
            yaxis_title="MAE",
            template="plotly_white"
        )

        st.plotly_chart(fig_comp, use_container_width=True)

    st.header("🔮 Forecast Future Prices")

    if st.button("🔮 Generate ARIMA Future Forecast"):
        dates, preds, error = forecast_arima(ts, forecast_steps)

        if error:
            st.error(f"Forecast failed: {error}")
        else:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=ts.index,
                y=ts["Price"],
                mode="lines",
                name="Historical"
            ))

            fig.add_trace(go.Scatter(
                x=dates,
                y=preds,
                mode="lines",
                name="Forecast",
                line=dict(dash="dash")
            ))

            fig.update_layout(
                title=f"{forecast_steps}-Day ARIMA Forecast",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model Explanations"):
        st.markdown("""
        - **Naive Baseline**: Predicts the current price using the previous available price.
        - **Moving Average 7**: Predicts using the average of the previous 7 prices.
        - **Random Forest**: Uses lag features, rolling mean, rolling volatility, and returns.
        - **XGBoost**: Gradient boosting model using the same engineered time-series features.
        - **ARIMA**: Statistical time-series model using past values and differencing.
        - **Directional Accuracy**: Measures whether the model correctly predicts price movement direction.
        """)

    st.markdown("---")
    st.markdown(
        "**Note:** In short-term financial forecasting, simple baselines can be very strong. "
        "Advanced models should always be compared against them."
    )


if __name__ == "__main__":
    main()