
import streamlit as st
import joblib
import pandas as pd
import re
import json

STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

def clean_text(s):
    """
    Clean the review text: lowercase, remove URLs, punctuation, extra spaces,
    stopwords, and short words.
    """
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return " ".join([w for w in s.split() if w not in STOP_WORDS and len(w) > 1])

@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        model = joblib.load("logistic_regression.joblib")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the model files are in the directory.")
        return None, None

@st.cache_data
def load_results():
    try:
        with open('results.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

st.set_page_config(page_title="Fake Review Detector", layout="wide")

st.markdown("# 🕵️ Fake Review Detector")
st.markdown("Upload a CSV with a column containing review text, or type/paste a review below. The model predicts whether the review is Genuine or Fake.")

col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
    review = st.text_area("Or paste/type a single review here:")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write('Preview:', df.head(3))
        text_cols = [c for c in df.columns if df[c].dtype == 'object']
        if text_cols:
            choice = st.selectbox("Choose text column", options=text_cols)
            if st.button("Predict on uploaded CSV"):
                vectorizer, model = load_models()
                if vectorizer is None or model is None:
                    st.error("Models could not be loaded.")
                else:
                    texts = df[choice].astype(str).apply(clean_text)
                    X = vectorizer.transform(texts)
                    preds = model.predict(X)
                    probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
                    df['prediction'] = preds
                    df['predicted_label'] = ['Genuine' if p == 0 else 'Fake' for p in preds]
                    if probs is not None:
                        df['prob_fake'] = probs
                    st.write(df.head(20))
                    st.success("Predictions added to preview. Download the result using the button below.")
                    st.download_button('Download predictions as CSV', df.to_csv(index=False).encode('utf-8'), 'predictions.csv')
with col2:
    st.markdown("### Model Info")
    results = load_results()
    if results:
        best_model = results.get('chosen_model', 'logistic_regression')
        st.write(f"Best model: **{best_model}**")
        st.write("Evaluation on holdout:")
        if 'logistic' in results:
            st.write(results['logistic'])
        else:
            st.write("Metrics not available.")
    else:
        st.write("Best model: **logistic_regression**")
        st.write("Evaluation metrics unavailable.")
    st.markdown("---")

if review:
    clean = clean_text(review)
    vectorizer, model = load_models()
    if vectorizer is None or model is None:
        st.error("Models could not be loaded.")
    else:
        X = vectorizer.transform([clean])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
        label = "Fake" if pred == 1 else "Genuine"
        st.markdown(f"### Prediction: **{label}**")
        if prob is not None:
            st.markdown(f"Probability (fake): **{prob:.3f}**")
        st.info("Tip: paste longer reviews for better predictions.")
