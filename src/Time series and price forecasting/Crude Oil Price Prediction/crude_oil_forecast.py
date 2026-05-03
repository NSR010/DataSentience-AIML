import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


OUTPUT_DIR = "output"
DATA_PATH = "Crude_Oil_Data.csv"

print("Libraries imported")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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


def evaluate_model(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }


def detect_columns(df):
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
            raise ValueError("No suitable numeric price column found.")

    return date_col, price_col


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Place your CSV in the working folder.")

    df = pd.read_csv(data_path)

    date_col, price_col = detect_columns(df)
    print("Using", date_col, "as date and", price_col, "as price")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col])
    df = df.reset_index(drop=True)

    ts = df[[date_col, price_col]].rename(
        columns={date_col: "Date", price_col: "Price"}
    )

    ts = ts.set_index("Date")

    # Do not force daily frequency using asfreq("D").
    # Market data usually skips weekends and holidays.
    # Filling those dates creates artificial prices.
    ts["Price"] = ts["Price"].interpolate().ffill().bfill()

    return ts


def save_line_plot(x, y, title, filename, label=None):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, label=label)

    if label:
        plt.legend()

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()


def save_prediction_plot(dates, y_true, y_pred, title, filename, pred_label):
    plt.figure(figsize=(12, 3))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label=pred_label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()


def create_lag_features(series, lags=5):
    df = pd.DataFrame({
        "Date": series.index,
        "Price": series.values,
    })

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Price"].shift(i)

    df["rolling_mean_7"] = df["Price"].rolling(7).mean()
    df["rolling_std_7"] = df["Price"].rolling(7).std()
    df["return_1d"] = df["Price"].pct_change(1)

    df = df.dropna().reset_index(drop=True)

    return df


def run_baselines(ts, test_size):
    test = ts.iloc[-test_size:].copy()
    y_true = test["Price"].values

    naive_pred = ts["Price"].shift(1).iloc[-test_size:].values
    ma7_pred = ts["Price"].rolling(7).mean().iloc[-test_size:].values

    naive_metrics = evaluate_model("Naive Baseline", y_true, naive_pred)
    ma7_metrics = evaluate_model("Moving Average 7", y_true, ma7_pred)

    print("Naive Baseline RMSE", naive_metrics["RMSE"], "MAE", naive_metrics["MAE"], "MAPE", naive_metrics["MAPE"])
    print("Moving Average 7 RMSE", ma7_metrics["RMSE"], "MAE", ma7_metrics["MAE"], "MAPE", ma7_metrics["MAPE"])

    save_prediction_plot(
        test.index,
        y_true,
        naive_pred,
        "Naive Baseline Predictions",
        "naive_predictions.png",
        "Naive Baseline",
    )

    save_prediction_plot(
        test.index,
        y_true,
        ma7_pred,
        "7-Day Moving Average Predictions",
        "moving_average_predictions.png",
        "MA7",
    )

    return [naive_metrics, ma7_metrics]


def run_random_forest(ts, test_size, lags=5):
    rf_df = create_lag_features(ts["Price"], lags=lags)

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)] + [
        "rolling_mean_7",
        "rolling_std_7",
        "return_1d",
    ]

    split = len(rf_df) - test_size

    X_train = rf_df[feature_cols].iloc[:split].values
    y_train = rf_df["Price"].iloc[:split].values

    X_test = rf_df[feature_cols].iloc[split:].values
    y_test = rf_df["Price"].iloc[split:].values

    print("RF train/test shapes:", X_train.shape, X_test.shape)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    metrics = evaluate_model("Random Forest", y_test, rf_pred)

    print("Random Forest RMSE", metrics["RMSE"], "MAE", metrics["MAE"], "MAPE", metrics["MAPE"])

    save_prediction_plot(
        rf_df["Date"].iloc[split:],
        y_test,
        rf_pred,
        "Random Forest Predictions",
        "rf_predictions.png",
        "Random Forest",
    )

    return metrics


def run_xgboost(ts, test_size, lags=5):
    if not XGBOOST_AVAILABLE:
        print("XGBoost not installed. Skipping XGBoost model.")
        return None

    xgb_df = create_lag_features(ts["Price"], lags=lags)

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)] + [
        "rolling_mean_7",
        "rolling_std_7",
        "return_1d",
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
        random_state=42,
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    metrics = evaluate_model("XGBoost", y_test, xgb_pred)

    print("XGBoost RMSE", metrics["RMSE"], "MAE", metrics["MAE"], "MAPE", metrics["MAPE"])

    save_prediction_plot(
        xgb_df["Date"].iloc[split:],
        y_test,
        xgb_pred,
        "XGBoost Predictions",
        "xgboost_predictions.png",
        "XGBoost",
    )

    return metrics


def run_arima(ts, train, test, order=(5, 1, 0)):
    try:
        arima_model = ARIMA(train["Price"], order=order)
        arima_res = arima_model.fit()

        arima_forecast = arima_res.forecast(steps=len(test))
        arima_pred = np.array(arima_forecast)

        metrics = evaluate_model("ARIMA", test["Price"].values, arima_pred)

        print("ARIMA RMSE", metrics["RMSE"], "MAE", metrics["MAE"], "MAPE", metrics["MAPE"])

        save_prediction_plot(
            test.index,
            test["Price"].values,
            arima_pred,
            "ARIMA Predictions",
            "arima_predictions.png",
            "ARIMA",
        )

        return metrics

    except Exception as e:
        print("ARIMA failed:", e)
        return None


def print_terminal_summary(results_df):
    print("\n" + "=" * 70)
    print("CRUDE OIL FORECASTING MODEL EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\nModels evaluated:")
    for model in results_df["Model"]:
        print(f"- {model}")

    print("\nModel ranking by MAE:")
    for idx, row in results_df.iterrows():
        print(
            f"{idx + 1}. {row['Model']} | "
            f"MAE: {row['MAE']:.4f} | "
            f"RMSE: {row['RMSE']:.4f} | "
            f"MAPE: {row['MAPE']:.2f}% | "
            f"Directional Accuracy: {row['Directional_Accuracy']:.4f}"
        )

    best_model = results_df.iloc[0]

    print("\nBest model based on MAE:")
    print(f"{best_model['Model']} with MAE = {best_model['MAE']:.4f}")

    print("\nWhy the naive baseline is included:")
    print(
        "The naive baseline predicts the current price using the previous available price. "
        "It is a simple but important benchmark because advanced models should ideally "
        "perform better than this basic persistence method."
    )

    print("\nOutput files generated:")
    print("- output/crude_oil_price.png")
    print("- output/rolling_mean.png")
    print("- output/rolling_std.png")
    print("- output/naive_predictions.png")
    print("- output/moving_average_predictions.png")
    print("- output/rf_predictions.png")
    print("- output/xgboost_predictions.png if XGBoost is installed")
    print("- output/arima_predictions.png if ARIMA succeeds")
    print("- output/model_comparison.csv")
    print("- output/model_comparison_mae.png")

    print("\nConclusion:")
    print(
        "This experiment compares baseline, machine learning, and statistical forecasting "
        "models using the same time-based test split and shared evaluation metrics."
    )

    print("=" * 70)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = load_data(DATA_PATH)

    save_line_plot(
        ts.index,
        ts["Price"],
        "Crude Oil Price",
        "crude_oil_price.png",
    )

    save_line_plot(
        ts.index,
        ts["Price"].rolling(30).mean(),
        "30-Day Rolling Mean",
        "rolling_mean.png",
    )

    save_line_plot(
        ts.index,
        ts["Price"].rolling(30).std(),
        "30-Day Rolling Std",
        "rolling_std.png",
    )

    adf_res = adfuller(ts["Price"].dropna())
    print("ADF statistic:", adf_res[0], "p-value:", adf_res[1])

    test_size = int(len(ts) * 0.2)
    train = ts.iloc[:-test_size].copy()
    test = ts.iloc[-test_size:].copy()

    print("Train rows", len(train), "Test rows", len(test))

    all_results = []

    baseline_results = run_baselines(ts, test_size)
    all_results.extend(baseline_results)

    rf_metrics = run_random_forest(ts, test_size, lags=5)
    all_results.append(rf_metrics)

    xgb_metrics = run_xgboost(ts, test_size, lags=5)
    if xgb_metrics is not None:
        all_results.append(xgb_metrics)

    arima_metrics = run_arima(ts, train, test, order=(5, 1, 0))
    if arima_metrics is not None:
        all_results.append(arima_metrics)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("MAE").reset_index(drop=True)

    print("\nModel Comparison:")
    print(results_df)

    print_terminal_summary(results_df)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(results_df["Model"], results_df["MAE"])
    plt.title("Model Comparison by MAE")
    plt.ylabel("MAE")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_mae.png"))
    plt.show()

    print("Saved plots and comparison table to ./output")


if __name__ == "__main__":
    main()