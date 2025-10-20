# utils.py -text preprocessing for VADER sentiment analysis
import re

def clean_text(text):
    """Simple text cleaning for Twitter data"""
    s = str(text)
    # Remove RT, URLs, mentions, hashtags
    s = s.replace("RT", "")
    s = re.sub(r'http\S+', '', s)  # remove urls
    s = re.sub(r'@\w+', '', s)     # remove mentions
    s = s.replace("#", " ")        # remove hashtag sign
    # Keep only letters, numbers, spaces, and basic punctuation
    s = re.sub(r'[^A-Za-z0-9\s.,!?]', ' ', s)
    # Clean up multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s