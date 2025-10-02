import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

print("Training intent recognition model...")

with open('augmented_data.json', 'r') as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Create a machine learning pipeline
model = make_pipeline(TfidfVectorizer(stop_words='english'), LinearSVC())
model.fit(patterns, tags)

# Save the trained model
with open('intent_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as intent_model.pkl")