"""
Train a numeric-feature-based spam classifier and save artifacts.

Usage:
    python train_model.py

Notes:
- Edit DATA_CSV, LABEL_COL, ID_COL if your dataset column names differ.
- Produces models/best_model.joblib and models/feature_columns.joblib and models/training_report.json
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

DATA_CSV = "./dataset.csv"   
LABEL_COL = "Prediction"          # default label column detected
ID_COL = "Email No."              # default id column detected

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data from:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in CSV. Columns: {df.columns.tolist()[:50]}")
    feature_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    print(f"Detected {len(feature_cols)} feature columns. Sample features:", feature_cols[:10])

    X = df[feature_cols].fillna(0).astype(float)
    y = df[LABEL_COL].astype(int)

    # Use stratified split to preserve label ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scaling + logistic regression baseline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=1000))
    ])

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating model on test set...")
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc = None

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save artifacts
    model_path = OUT_DIR / "best_model.joblib"
    features_path = OUT_DIR / "feature_columns.joblib"
    report_path = OUT_DIR / "training_report.json"

    joblib.dump(pipe, model_path)
    joblib.dump(feature_cols, features_path)
    summary = {
        "num_rows": int(df.shape[0]),
        "num_features": len(feature_cols),
        "feature_sample": feature_cols[:50],
        "accuracy": acc,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": report
    }
    report_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved model -> {model_path}")
    print(f"Saved features -> {features_path}")
    print(f"Saved training report -> {report_path}")
    print("Done.")

if __name__ == "__main__":
    main()
