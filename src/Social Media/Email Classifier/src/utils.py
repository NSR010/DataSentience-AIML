"""
Helper functions to load the trained pipeline and perform predictions.

Functions:
- load_model(): returns loaded pipeline and feature_columns
- predict_row(row_dict) -> dict: accept dict or pandas Series; returns {label, probability}
- prepare_row(row): ensure feature order, fill missing values
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.joblib"

def load_model():
    """Load model pipeline and feature columns. Raises if not found."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Model or features not found. Expected at: {MODEL_PATH} and {FEATURES_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols

def prepare_row(row, feature_cols):
    """
    Given a dict-like or pandas Series, return numpy array matching feature_cols.
    Missing features are filled with 0.
    """
    # Accept pandas Series, dict, or list-like
    if isinstance(row, pd.Series):
        s = row
    elif isinstance(row, dict):
        s = pd.Series(row)
    else:
        s = pd.Series(row)
    # Reindex to feature cols and fillna with 0
    s = s.reindex(feature_cols).fillna(0).astype(float)
    return s.values.reshape(1, -1)

def predict_row(row):
    """
    Predict on a single row (dict-like or pandas Series).
    Returns: {"label": int, "probability": float}
    """
    model, feature_cols = load_model()
    X = prepare_row(row, feature_cols)
    pred = int(model.predict(X)[0])
    prob = None
    try:
        prob = float(model.predict_proba(X)[0].max())
    except Exception:
        prob = None
    return {"label": pred, "probability": prob}

def predict_batch(df):
    """
    Predict on a pandas DataFrame that contains the expected feature columns.
    Returns predictions as a pandas Series.
    """
    model, feature_cols = load_model()
    if not set(feature_cols).issubset(set(df.columns)):
        missing = set(feature_cols) - set(df.columns)
        raise ValueError(f"Missing expected feature columns: {list(missing)[:20]}")
    X = df[feature_cols].fillna(0).astype(float)
    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = None
    out = df.copy()
    out["prediction"] = preds
    if probs is not None:
        out["probability"] = probs
    return out
