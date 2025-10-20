# Email Spam Classification Project

This project implements a machine learning-based email spam classifier using numeric features derived from email text (e.g., word and character frequencies). It includes training a model, a Streamlit web app for predictions, and utilities for batch and single-row predictions.

## Features

- **Model Training**: Trains a Logistic Regression model with StandardScaler on precomputed numeric features.
- **Streamlit App**: Interactive web interface for:
  - Batch predictions via CSV upload.
  - Single-row manual predictions with a subset of features.
- **Model Artifacts**: Saves trained model, feature columns, and training report.
- **High Accuracy**: Achieves ~96% accuracy on test set with good ROC-AUC.

## Project Structure

```
.
├── app.py                 # Streamlit web app for predictions
├── train_model.py         # Script to train the model
├── src/
│   └── utils.py           # Helper functions for loading model and predictions
├── models/                # Directory for saved model artifacts
│   ├── best_model.joblib      # Trained pipeline
│   ├── feature_columns.joblib # List of feature columns
│   └── training_report.json   # Training metrics and report
├── dataset.csv            # Training dataset (numeric features)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── TODO.md                # Task list (if present)
```

## Installation

1. Clone or download the project.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Ensure `dataset.csv` is in the project root. The dataset should have numeric feature columns, an ID column ("Email No."), and a label column ("Prediction").
2. Run the training script:
   ```
   python train_model.py
   ```
   This will create/update `models/best_model.joblib`, `models/feature_columns.joblib`, and `models/training_report.json`.

### Running the Streamlit App

1. After training, launch the app:
   ```
   streamlit run app.py
   ```
2. Open the provided URL in your browser.
3. Use the interface for batch CSV predictions or manual single-row inputs.

### Batch Predictions

- Upload a CSV with the same feature columns as training data.
- The app checks for missing features and predicts labels/probabilities.

### Single-Row Predictions

- Manually input values for the first 50 features (expandable in code).
- Get instant prediction results.

## Dataset

- **Source**: Assumed to be a preprocessed email dataset with numeric features (e.g., frequencies of words like "the", "to", etc.).
- **Columns**: ID ("Email No."), Label ("Prediction"), and 3000+ numeric features.
- **Size**: ~5172 rows.

## Model Details

- **Algorithm**: Logistic Regression with StandardScaler.
- **Evaluation**: Stratified train-test split (80-20).
- **Metrics** (from latest training):
  - Accuracy: ~96%
  - ROC-AUC: ~98%
  - Precision/Recall/F1: High for both classes.

## Customization

- **Change Dataset**: Edit `DATA_CSV` in `train_model.py`.
- **Adjust Columns**: Modify `LABEL_COL` and `ID_COL` in `train_model.py`.
- **Model Tuning**: Update the pipeline in `train_model.py` (e.g., change classifier or hyperparameters).
- **App Features**: Edit `app.py` to add more features or change UI.

## Requirements

- Python 3.7+
- Libraries: streamlit, scikit-learn, pandas, numpy, joblib

## Notes

- The model expects numeric features; for text-based classification, implement TF-IDF preprocessing.
- Ensure uploaded CSVs match the feature columns exactly.
- For production, consider model versioning and deployment (e.g., via FastAPI).

