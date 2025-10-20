import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from utils import clean_text
import joblib

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def get_vader_features(text):
    """Extract VADER sentiment features from text"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return [
        scores['compound'],  # Overall sentiment (-1 to 1)
        scores['pos'],       # Positive score (0 to 1)
        scores['neg'],       # Negative score (0 to 1)
        scores['neu']        # Neutral score (0 to 1)
    ]

def main():
    print("Loading data...")
    # Load training data
    df = pd.read_csv('dataset.csv', header=None, names=['id','topic','sentiment','text'])
    df = df.dropna(subset=['text','sentiment'])
    
    # Clean and standardize sentiment labels
    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
    df = df[df['sentiment'].isin(['Positive','Negative','Neutral'])]
    df['label'] = df['sentiment'].map({'Positive':'positive','Negative':'negative','Neutral':'neutral'})
    
    print(f"Loaded {len(df)} tweets")
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Clean text
    print("Cleaning text...")
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    
    # Extract VADER features
    print("Extracting VADER features...")
    vader_features = df['clean_text'].apply(get_vader_features)
    X = np.array(vader_features.tolist())
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train simple Random Forest
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'simple_model.joblib')
    print("\nModel saved as 'simple_model.joblib'")
    
    # Show feature importance
    feature_names = ['compound', 'positive', 'negative', 'neutral']
    importances = model.feature_importances_
    print("\nFeature importance:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.3f}")

if __name__ == '__main__':
    main()
