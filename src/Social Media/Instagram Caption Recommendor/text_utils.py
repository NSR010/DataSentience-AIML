import random
from googletrans import Translator

translator = Translator()

EMOJIS = {
    "positive": ["✨", "🔥", "❤️", "😍", "🌅", "🌿", "😊", "🤍", "🤳", "📸"],
    "funny": ["😂", "🤣", "😅", "🙈", "😜"],
    "inspirational": ["💪", "🌟", "🚀", "🌈", "✨"],
    "romantic": ["💕", "❤️", "🌹", "💑", "🌙"],
    "motivational": ["💪", "🔥", "🚀", "🌟", "⚡"],
    "default": ["✨", "🔥", "❤️", "😍", "🌅", "🌿", "😅", "🤍", "🤳", "📸"]
}

POSITIVE_WORDS = ["good", "great", "awesome", "amazing", "love", "happy", "beautiful", "excited", "wonderful", "fantastic"]
NEGATIVE_WORDS = ["bad", "sad", "angry", "hate", "terrible", "awful", "disappointed", "frustrated", "horrible", "worst"]

def analyze_sentiment(text: str) -> str:
    """
    Simple sentiment analysis using keyword matching.

    Returns:
        'positive', 'negative', or 'neutral'
    """
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def insert_emojis(text: str, tone: str = "default") -> str:
    """
    Insert relevant emojis based on tone and sentiment.

    Args:
        text: Caption text
        tone: Tone of the caption

    Returns:
        Text with emoji added
    """
    sentiment = analyze_sentiment(text)
    emoji_list = EMOJIS.get(tone, EMOJIS["default"])

    if sentiment == "positive" and tone in ["inspirational", "motivational"]:
        emoji_list = EMOJIS.get(tone, EMOJIS["positive"])

    if random.random() > 0.3:  
        return text + " " + random.choice(emoji_list)
    return text

def apply_tone(text: str, tone: str) -> str:
    """
    Apply tone modifications to the text.

    Args:
        text: Original text
        tone: Desired tone

    Returns:
        Modified text
    """
    if tone == "funny":
        funny_additions = [" 😂", " lol", " haha", " 🤣"]
        return text + random.choice(funny_additions)
    elif tone == "formal":
        formal_text = text.replace("I'm", "I am").replace("can't", "cannot").replace("won't", "will not")
        return formal_text + "."
    elif tone == "inspirational":
        inspirational_additions = [" ✨", " 🌟", " 💫", " Keep believing!"]
        return text + random.choice(inspirational_additions)
    elif tone == "sarcastic":
        return text + " 🙄"
    return text

def apply_length(text: str, length: str) -> str:
    """
    Adjust text length based on preference.

    Args:
        text: Original text
        length: 'short', 'medium', or 'long'

    Returns:
        Adjusted text
    """
    words = text.split()
    if length == "short" and len(words) > 6:
        return " ".join(words[:6]) + "..."
    elif length == "long":
        extensions = [
            " Remember, every moment matters.",
            " This is just the beginning.",
            " Embrace the journey.",
            " Life is full of these beautiful moments."
        ]
        return text + random.choice(extensions)
    return text

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to target language with error handling.

    Args:
        text: Text to translate
        target_lang: Target language code

    Returns:
        Translated text or original if translation fails
    """
    if target_lang == "none":
        return text

    try:
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text on failure
