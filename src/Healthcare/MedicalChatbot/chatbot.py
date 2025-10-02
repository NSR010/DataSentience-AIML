import json
import pickle
import random

# Load the trained intent model
with open('intent_model.pkl', 'rb') as f:
    intent_model = pickle.load(f)

# Load the responses
with open('data.json', 'r') as f:
    data = json.load(f)
    responses = {intent['tag']: intent['responses'] for intent in data['intents']}

# Expanded keywords for simple entity extraction
SYMPTOMS = [
    "headache", "pain", "chest", "back", "dizzy", "fever", "cough", "sore", "nauseous",
    "throat", "swollen", "tired", "sneezing", "cold", "stomach", "shoulder", "ear",
    "itchy", "skin", "burn", "sprain", "cramps", "pressure"
]
DURATIONS = ["hour", "hours", "day", "days", "week", "weeks", "month", "months"]
SEVERITY = ["mild", "moderate", "severe", "extreme", "worsening"]

def extract_entities(text):
    entities = {"symptoms": [], "duration": [], "severity": []}
    words = text.lower().split()
    for word in words:
        if word in SYMPTOMS:
            entities["symptoms"].append(word)
        if word in DURATIONS:
            entities["duration"].append(word)
        if word in SEVERITY:
            entities["severity"].append(word)
    return entities

def get_recommendation(entities):
    # Emergency check
    if "chest" in entities["symptoms"] and "pain" in entities["symptoms"]:
        return "⚠️ Chest pain can be serious. Please seek immediate emergency care."
    if "faint" in entities["symptoms"] or "shortness of breath" in entities["symptoms"]:
        return "⚠️ This seems urgent. Please seek emergency help immediately."

    # Common symptom checks
    if "headache" in entities["symptoms"]:
        return "For a headache, rest and hydration can help. If severe or persistent, see a doctor."
    if "fever" in entities["symptoms"]:
        return "For fever, stay hydrated and rest. If it lasts more than 3 days or is very high, consult a doctor."
    if "cough" in entities["symptoms"] or "throat" in entities["symptoms"]:
        return "A sore throat or cough may improve with warm fluids and rest. If it worsens, consider seeing a doctor."

    # Default fallback
    return "Please monitor your condition and consult a healthcare professional if it worsens."

def chat():
    print("🩺 Medical Chatbot is online! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Bot:", random.choice(responses["goodbye"]))
            break
        
        # Predict intent
        intent = intent_model.predict([user_input])[0]

        if intent in responses:
            print("Bot:", random.choice(responses[intent]))
        else:
            print("Bot: I'm not sure how to respond to that.")

        # If reporting a symptom, extract details
        if intent == "report_symptom":
            entities = extract_entities(user_input)
            if any(entities.values()):
                print("Bot Recommendation:", get_recommendation(entities))
            else:
                print("Bot: Can you tell me how long you've been experiencing this and how severe it feels?")
        
        elif intent == "symptom_duration":
            print("Bot: Thanks for sharing the duration. Does it feel mild, moderate, or severe?")
        
        elif intent == "symptom_severity":
            print("Bot: Thanks for clarifying severity. I’ll keep that in mind.")
        
        elif intent == "emergency":
            print("⚠️ Bot: This sounds very serious. Please seek emergency care immediately.")

if __name__ == "__main__":
    chat()
    