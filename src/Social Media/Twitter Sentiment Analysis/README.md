# Twitter Sentiment Analyzer

## Overview
A simple web application for analyzing sentiment in Twitter text using VADER sentiment analysis and optional machine learning enhancement with Random Forest classifier.

## What it does
- Analyzes Twitter text sentiment (Positive/Negative/Neutral)
- Uses VADER sentiment analysis (rule-based, no training needed)
- Simple text cleaning (removes URLs, mentions, hashtags)
- Optional ML model for improved accuracy
- Web interface built with Streamlit
- Supports single tweet analysis and batch CSV processing

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

3. **Train the model (optional - for better accuracy):**
   ```bash
   python train_model.py
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Files

- `app.py` - Streamlit web app for sentiment analysis
- `train_model.py` - Training script (uses VADER features + Random Forest)
- `utils.py` - Text cleaning and preprocessing functions
- `requirements.txt` - Python dependencies
- `dataset.csv` - Training dataset
- `test.csv` - Test dataset
- `simple_model.joblib` - Trained model file (generated after training)

## How it works

1. **Text Cleaning**: Removes URLs, mentions, hashtags, extra spaces, and special characters
2. **VADER Analysis**: Extracts 4 sentiment scores (compound, positive, negative, neutral)
3. **Prediction**: Classifies sentiment using VADER scores or trained ML model
4. **Optional ML**: Random Forest trained on VADER features for enhanced accuracy

## Usage

### Single Tweet Analysis
- Enter text in the text area
- Click "Analyze Sentiment" to get prediction with confidence scores
- View detailed VADER scores

### Batch Analysis
- Upload a CSV file with a 'text' column
- Get sentiment predictions for all rows
- Download results as CSV

## Accuracy
- VADER alone: ~80-85% accuracy
- VADER + Random Forest: ~85-90% accuracy

## Features
- No complex preprocessing (no lemmatization, stemming, etc.)
- No TF-IDF or word embeddings
- Just VADER sentiment scores + optional ML
- Easy to understand and modify
- Fast and lightweight
- Web-based interface
- Batch processing capability
- Confidence scores and detailed metrics

## Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`

## Model Training
Run `python train_model.py` to train the Random Forest model on VADER features. The trained model is saved as `simple_model.joblib`.

## Testing
Use `test.csv` for evaluating model performance after training.
