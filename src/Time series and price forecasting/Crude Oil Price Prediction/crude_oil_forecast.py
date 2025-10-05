import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import joblib
import warnings
warnings.filterwarnings('ignore')

print('Libraries imported')

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

data_path = 'Crude_Oil_Data.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Place your CSV in the working folder.")

df = pd.read_csv(data_path)
df.columns
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

print('Using', date_col, 'as date and', price_col, 'as price')
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
ts = df[[date_col, price_col]].rename(columns={date_col:'Date', price_col:'Price'}).set_index('Date')
ts = ts.asfreq('D')
ts['Price'] = ts['Price'].interpolate(method='time').ffill().bfill()
ts.head()

os.makedirs('output', exist_ok=True)

plt.figure(figsize=(12,4))
plt.plot(ts['Price'])
plt.title('Crude Oil Price')
plt.savefig(os.path.join('output', 'crude_oil_price.png'))
plt.show()

plt.figure(figsize=(12,3))
plt.plot(ts['Price'].rolling(30).mean())
plt.title('30-day rolling mean')
plt.savefig(os.path.join('output', 'rolling_mean.png'))
plt.show()

plt.figure(figsize=(12,3))
plt.plot(ts['Price'].rolling(30).std())
plt.title('30-day rolling std')
plt.savefig(os.path.join('output', 'rolling_std.png'))
plt.show()

adf_res = adfuller(ts['Price'].dropna())
print('ADF statistic:', adf_res[0], 'p-value:', adf_res[1])

test_size = int(len(ts) * 0.2)
train = ts.iloc[:-test_size].copy()
test = ts.iloc[-test_size:].copy()
print('Train rows', len(train), 'Test rows', len(test))

def create_lag_features(series, lags=5):
    df = pd.DataFrame({'Date':series.index, 'Price':series.values})
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['Price'].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df

lags = 5
rf_df = create_lag_features(ts['Price'], lags=lags)
total = len(ts)
train_size = total - test_size
rf_train = rf_df.iloc[:max(0, train_size-lags)].reset_index(drop=True)
rf_test = rf_df.iloc[max(0, train_size-lags): max(0, train_size-lags) + test_size].reset_index(drop=True)
X_train = rf_train[[f'lag_{i}' for i in range(1,lags+1)]].values
y_train = rf_train['Price'].values
X_test = rf_test[[f'lag_{i}' for i in range(1,lags+1)]].values
y_test = rf_test['Price'].values
print('RF train/test shapes:', X_train.shape, X_test.shape)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mape = mape(y_test, rf_pred)
print('RF RMSE', rf_rmse, 'MAE', rf_mae, 'MAPE', rf_mape)

plt.figure(figsize=(12,3)); plt.plot(rf_test['Date'], y_test, label='Actual'); plt.plot(rf_test['Date'], rf_pred, label='RF'); plt.legend(); plt.savefig(os.path.join('output', 'rf_predictions.png')); plt.show()

arima_order = (5,1,0)

try:
    arima_model = ARIMA(train['Price'], order=arima_order)
    arima_res = arima_model.fit()
    arima_forecast = arima_res.forecast(steps=len(test))
    arima_pred = np.array(arima_forecast)
    arima_rmse = mean_squared_error(test['Price'], arima_pred, squared=False)
    arima_mae = mean_absolute_error(test['Price'], arima_pred)
    arima_mape = mape(test['Price'].values, arima_pred)
    print('ARIMA RMSE', arima_rmse, 'MAE', arima_mae, 'MAPE', arima_mape)
    plt.figure(figsize=(12,3)); plt.plot(test.index, test['Price'], label='Actual'); plt.plot(test.index, arima_pred, label='ARIMA'); plt.legend(); plt.savefig(os.path.join('output', 'arima_predictions.png')); plt.show()

except Exception as e:
    print('ARIMA failed:', e)

'''

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping



scaler = MinMaxScaler()

scaled = scaler.fit_transform(ts[['Price']])

lookback = 60

def create_sequences(data, lookback=60):

    X, y = [], []

    for i in range(lookback, len(data)):

        X.append(data[i-lookback:i, 0])

        y.append(data[i, 0])

    X, y = np.array(X), np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y



X_all, y_all = create_sequences(scaled, lookback=lookback)

train_end_idx = len(train)

seq_split = train_end_idx - lookback

X_train_lstm = X_all[:seq_split]

y_train_lstm = y_all[:seq_split]

X_test_lstm = X_all[seq_split:seq_split+len(test)]

y_test_lstm = y_all[seq_split:seq_split+len(test)]



model = Sequential([LSTM(64, input_shape=(X_train_lstm.shape[1], 1)), Dropout(0.2), Dense(1)])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train_lstm, y_train_lstm, validation_split=0.1, epochs=20, batch_size=32, callbacks=[es])

pred_scaled = model.predict(X_test_lstm).flatten()

pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

'''

print('Saved plots to ./output')
