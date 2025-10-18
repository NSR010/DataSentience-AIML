# Fake Review Detector

This project trains models to detect fake reviews using TF-IDF and classic ML models (Logistic Regression, Random Forest).
Files included:
- app.py : Streamlit app to interact with the model
- tfidf_vectorizer.joblib : fitted TF-IDF vectorizer
- logistic_regression.joblib : best model selected by F1 on holdout
- logistic_model.joblib, random_forest_model.joblib : both trained models
- dataset_preview.json : preview of uploaded dataset
- results.json : evaluation metrics for both models

Author - ANU

How to run:
1. Install requirements: `pip install -r requirements.txt`
2. Train the model: `python train_model.py`
3. Test the model: `python test_model.py`
4. Run streamlit: `streamlit run app.py`

Notes:
- This project uses a built-in stopword list to avoid external downloads in restricted environments.
