from collections import Counter
import random

nlp = None  

TRENDING_HASHTAGS = [
    "#fyp", "#viral", "#explore", "#love", "#instagood", "#photooftheday",
    "#beautiful", "#happy", "#cute", "#fashion", "#followme", "#picoftheday",
    "#art", "#photography", "#nature", "#travel", "#fitness", "#beauty"
]

def make_hashtags(caption: str, max_tags: int = 8, include_trending=False):
    """
    Generate hashtags from caption with optional trending hashtags.

    Args:
        caption: Caption text
        max_tags: Maximum number of hashtags to generate
        include_trending: Whether to include trending hashtags

    Returns:
        List of hashtags
    """
    if not nlp:
        words = [w.strip(".,!?").lower() for w in caption.split() if len(w) > 3]
        common = list(dict.fromkeys(words))[:max_tags]
    else:
        doc = nlp(caption)
        nouns = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
        common = [w for w, _ in Counter(nouns).most_common(max_tags)]

    hashtags = [f"#{w}" for w in common]

    if include_trending:
        trending = random.sample(TRENDING_HASHTAGS, min(3, len(TRENDING_HASHTAGS)))
        hashtags.extend(trending)

    return hashtags
