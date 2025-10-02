import json
import nltk
from nltk.corpus import wordnet
import random

# Download WordNet data (only needs to be done once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def get_synonyms(word):
    """Get synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def augment_sentence(sentence):
    """Create a new sentence by replacing one word with a synonym."""
    words = sentence.split()
    if len(words) < 3:
        return None # Don't augment very short sentences

    # Find a random word to replace that is not a stopword
    random_word_index = random.randint(0, len(words) - 1)
    random_word = words[random_word_index]
    
    synonyms = get_synonyms(random_word)
    if not synonyms:
        return None # No synonyms found

    new_word = random.choice(synonyms)
    new_sentence = ' '.join([new_word if i == random_word_index else w for i, w in enumerate(words)])
    return new_sentence

# --- Main Script ---
print("Starting data augmentation...")

# 1. Load the original data
with open('data.json', 'r') as f:
    data = json.load(f)

new_data = {"intents": []}

# 2. Loop through each intent and its patterns
for intent in data['intents']:
    new_patterns = list(intent['patterns']) # Start with original patterns
    
    # 3. Generate augmented sentences for each original pattern
    for pattern in intent['patterns']:
        augmented = augment_sentence(pattern)
        if augmented:
            new_patterns.append(augmented)
            print(f"Original: '{pattern}' -> Augmented: '{augmented}'")

    # 4. Create a new intent object with the expanded list of patterns
    new_intent_obj = {
        "tag": intent['tag'],
        "patterns": list(set(new_patterns)), # Use set to remove duplicates
        "responses": intent['responses']
    }
    new_data['intents'].append(new_intent_obj)

# 5. Save the new, augmented data to a different file
with open('augmented_data.json', 'w') as f:
    json.dump(new_data, f, indent=2)

print("\nAugmentation complete! New data saved to 'augmented_data.json'.")