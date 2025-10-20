import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from utils import clean_text

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title='Simple Sentiment Analyzer', layout='wide')
st.title('Simple Twitter Sentiment Analyzer')
st.markdown('Using VADER sentiment analysis - no complex ML needed!')

@st.cache_resource
def load_model():
    try:
        return joblib.load('simple_model.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def get_vader_features(text):
    """Extract VADER sentiment features from text"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return [scores['compound'], scores['pos'], scores['neg'], scores['neu']]

def predict_sentiment(text):
    """Predict sentiment using VADER features"""
    model = load_model()
    if model is None:
        return "Model not found. Please train first using train_simple.py"
    
    cleaned_text = clean_text(text)
    features = get_vader_features(cleaned_text)
    prediction = model.predict([features])[0]
    
    # Get confidence scores
    probabilities = model.predict_proba([features])[0]
    classes = model.classes_
    confidence = max(probabilities)
    
    return prediction, confidence, dict(zip(classes, probabilities))

# Load model
model = load_model()
if model is None:
    st.error("Model not found! Please run 'python train_simple.py' first to train the model.")
    st.stop()

# Single tweet prediction
st.header("Single Tweet Analysis")
tweet_text = st.text_area("Enter your tweet:", height=100)

if st.button("Analyze Sentiment"):
    if not tweet_text.strip():
        st.warning("Please enter some text first!")
    else:
        prediction, confidence, all_scores = predict_sentiment(tweet_text)
        
        # Display result
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction")
            if prediction == 'positive':
                st.success(f"😊 Positive ({confidence:.1%} confidence)")
            elif prediction == 'negative':
                st.error(f"😞 Negative ({confidence:.1%} confidence)")
            else:
                st.info(f"😐 Neutral ({confidence:.1%} confidence)")
        
        with col2:
            st.subheader("Detailed Scores")
            for sentiment, score in all_scores.items():
                st.write(f"{sentiment.capitalize()}: {score:.1%}")

# Batch prediction
st.header("Batch Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column!")
        else:
            st.write(f"Processing {len(df)} tweets...")
            
            # Process each tweet
            results = []
            for idx, row in df.iterrows():
                text = str(row['text'])
                cleaned = clean_text(text)
                prediction, confidence, _ = predict_sentiment(cleaned)
                results.append({
                    'original_text': text[:100] + '...' if len(text) > 100 else text,
                    'predicted_sentiment': prediction,
                    'confidence': f"{confidence:.1%}"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Show VADER scores for current text
if tweet_text:
    st.header("VADER Sentiment Scores")
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(clean_text(tweet_text))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Compound", f"{scores['compound']:.3f}")
    with col2:
        st.metric("Positive", f"{scores['pos']:.3f}")
    with col3:
        st.metric("Negative", f"{scores['neg']:.3f}")
    with col4:
        st.metric("Neutral", f"{scores['neu']:.3f}")
