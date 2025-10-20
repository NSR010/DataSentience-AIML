import joblib
import re
import pandas as pd

STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return " ".join([w for w in s.split() if w not in STOP_WORDS and len(w) > 1])

vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("logistic_regression.joblib")

# Load dataset
df = pd.read_csv('fake reviews dataset.csv')
df['label_binary'] = df['label'].map({'CG': 0, 'OR': 1})

# Sample some genuine and fake
genuine = df[df['label_binary'] == 0]['text_'].head(5).tolist()
fake = df[df['label_binary'] == 1]['text_'].head(5).tolist()

print("Testing Genuine Reviews:")
for text in genuine:
    cleaned = clean_text(text)
    print(f"Original: {text[:50]}...")
    print(f"Cleaned: '{cleaned[:50]}...'")
    if cleaned:
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        label = "Fake" if pred == 1 else "Genuine"
        print(f"Prediction: {label} (Prob Fake: {prob[1]:.3f})")
    print("---")

print("\nTesting Fake Reviews:")
for text in fake:
    cleaned = clean_text(text)
    print(f"Original: {text[:50]}...")
    print(f"Cleaned: '{cleaned[:50]}...'")
    if cleaned:
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        label = "Fake" if pred == 1 else "Genuine"
        print(f"Prediction: {label} (Prob Fake: {prob[1]:.3f})")
    print("---")
