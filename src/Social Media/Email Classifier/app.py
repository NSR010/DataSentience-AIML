"""
Streamlit app for the numeric-feature spam classifier.

Features:
- Upload CSV (with same feature columns) and batch predict
- Manual single-row prediction using a subset of features (first N)
- Shows basic metrics on batch predictions
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.utils import load_model, predict_row, predict_batch
import joblib

st.set_page_config(page_title="Email Spam Detector (Numeric)", layout="wide")
st.title("📧 Email Spam Classifier — Numeric Feature Model")
st.write("This demo expects a CSV with the same precomputed numeric features used in training (e.g., word/char frequencies).")

# Load model metadata (for UI)
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Model artifacts not found. Please run `python train_model.py` first to create models/best_model.joblib and models/feature_columns.joblib.")
    st.stop()

model, feature_cols = load_model()  # loads model and feature list
st.sidebar.header("Model Info")
st.sidebar.write(f"Features: {len(feature_cols)}")
st.sidebar.write("Model: StandardScaler + LogisticRegression (baseline)")

# Upload CSV for batch predictions
st.header("1) Batch prediction (CSV upload)")
uploaded = st.file_uploader("Upload CSV with feature columns (same as training features)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded file:", df.head())
        # Check missing features
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.warning(f"The uploaded file is missing {len(missing)} expected features. First 10 missing: {missing[:10]}")
        else:
            if st.button("Run batch prediction"):
                with st.spinner("Predicting..."):
                    out = predict_batch(df)
                st.write("Predicted sample rows:")
                st.dataframe(out.head(30))
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")
    except Exception as e:
        st.error(f"Failed to read or predict on uploaded CSV: {e}")

st.markdown("---")
# Single row manual prediction
st.header("2) Single-row prediction (manual input)")
st.write("Manually enter numeric feature values for a single email. For UI simplicity we show the first 50 features; you can expand code if you need all features.")

NUM_MANUAL = 50  # show first N features for manual entry 
manual_feature_subset = feature_cols[:NUM_MANUAL]

with st.form("manual_predict"):
    st.subheader("Enter numeric values (leave blank => 0)")
    cols = st.columns(2)
    inputs = {}
    for i, feat in enumerate(manual_feature_subset):
        col = cols[i % 2]
        inputs[feat] = col.number_input(feat, value=0.0, format="%.6f", key=f"f_{i}")

    submit = st.form_submit_button("Predict single row")
    if submit:
        with st.spinner("Predicting..."):
            res = predict_row(inputs)
        st.success(f"Prediction: {res['label']} (0 = not spam, 1 = spam)")
        if res.get("probability") is not None:
            st.write(f"Model probability (max class): {res['probability']:.4f}")

st.markdown("---")
st.header("Model & Data Info")
st.write(f"Number of features (expected): **{len(feature_cols)}**")
st.write("If your dataset originally had text (subject/body), you can build a TF-IDF pipeline and retrain a text-based classifier. This app currently expects numeric feature columns like word/char frequencies.")
st.write("To retrain the model on a different CSV, adjust `DATA_CSV` in `train_model.py` and run `python train_model.py`.")
