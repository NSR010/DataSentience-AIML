# Crude Oil Price Forecasting Dashboard

This project provides an interactive Streamlit dashboard and modular Python scripts for analyzing, modeling, and forecasting crude oil prices using machine learning and statistical models.
Author - ANU

## Features

- **Data Loading & Preprocessing:** Automatically detects date and price columns, handles missing data with interpolation.
- **Exploratory Data Analysis (EDA):** Visualize price trends, rolling means, and volatility with interactive Plotly charts.
- **Stationarity Testing:** Augmented Dickey-Fuller test to assess time series stationarity.
- **Machine Learning Models:**
  - Random Forest regression with lag features for price prediction.
  - ARIMA statistical model for time series forecasting.
- **Forecasting:** Generate future price forecasts with ARIMA and visualize alongside historical data.
- **User Controls:** Adjustable test size, lag features, and forecast horizon via sidebar sliders.
- **Robust Error Handling:** Informative UI messages for missing data, model errors, and invalid inputs.

## Getting Started

1. Place your dataset file `Crude_Oil_Data.csv` in the project root directory.

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Main Streamlit dashboard application.
- `crude_oil_forecast.py` - Standalone script for data loading, EDA, modeling, and forecasting.
- `README.md` - Project overview and instructions.
- `Crude_Oil_Data.csv` - Input dataset (user-provided).
- `output/` - Directory where generated plots are saved.

## Future Enhancements

- Incorporate advanced feature engineering (rolling windows, calendar effects, exogenous variables).
- Hyperparameter tuning for improved model accuracy.
- Add deep learning models such as LSTM or Transformer for enhanced forecasting.
- Expand dashboard with additional visualizations and user interactivity.
- Deploy as a web service for real-time crude oil price predictions.

## Notes

- The LSTM model is not included due to TensorFlow complexity but can be added in a TF-enabled environment.
- The dashboard uses Plotly for interactive charts and Matplotlib for static plots.
- Ensure your dataset contains appropriate date and price columns for smooth operation.

---

Feel free to explore, modify, and extend this project to suit your forecasting needs!
