import random
import json
import os
from text_utils import apply_tone, apply_length, insert_emojis

TEMPLATES = {
    "witty": [
        "{} — who needs filters?",
        "Guess what? {}",
        "Plot twist: {}",
        "Not me saying {} but...",
        "{} level: Expert",
        "POV: You're {}",
        "Savage mode: {}",
        "{} but make it fashion",
        "Adulting level: {}",
    ],
    "inspirational": [
        "{}. Keep shining!",
        "{} — chase your dreams.",
        "{} and believe in yourself.",
        "Embrace the {} within you.",
        "{}: Your journey starts now.",
        "Rise above and conquer {}.",
        "Dream big, achieve {}.",
        "{} is your superpower.",
        "Rise and shine with {}.",
    ],
    "aesthetic": [
        "{} • softly curated",
        "The vibe is {}",
        "{} — like a mood board",
        "Aesthetic: {}",
        "{} in soft pastels",
        "Minimalist {} vibes",
        "Soft {} dreams",
        "{} in golden hour",
        "Boho {} aesthetic",
    ],
    "trending": [
        "{} #foryou #viral",
        "{} — let's make it trend",
        "{} ✨🔥",
        "Trending: {}",
        "{} is the new black",
        "Everyone's doing {} now",
        "{} challenge accepted",
        "Viral {} incoming",
        "{} taking over",
    ],
    "romantic": [
        "{} with all my heart 💕",
        "Falling for {} every day",
        "{} and you — perfect match",
        "Love is {} and more",
        "Whispers of {} in the air",
        "{} under the stars",
        "Forever and {} always",
        "My heart says {}",
        "{} in your eyes",
    ],
    "motivational": [
        "Push through: {}",
        "{} — your strength lies here",
        "Conquer {} like a champion",
        "Level up with {}",
        "{}: Fuel for your fire",
        "Unstoppable {} mode activated",
        "Rise above the {}",
        "Champion mindset: {}",
        "Victory through {}",
    ],
    "poetic": [
        "In the garden of {}, dreams bloom",
        "{} whispers secrets to the wind",
        "Like {} in morning light",
        "Echoes of {} in my soul",
        "{} painted in verses",
        "Where {} meets eternity",
        "A symphony of {}",
        "Dancing with {} shadows",
    ],
    "sarcastic": [
        "Oh great, more {} 🙄",
        "{} because adulting is hard",
        "Totally not obsessed with {}",
        "{} level: Expert procrastinator",
        "Living my best {} life (said no one ever)",
        "Because {} wasn't enough drama",
        "Me: {} is life. Also me: *cries*",
    ],
    "adventurous": [
        "Conquering {} one step at a time",
        "Adventure awaits in {}",
        "Exploring the wild side of {}",
        "Thrill seeker in {}",
        "Chasing horizons with {}",
        "Boldly going where {} takes me",
        "Wanderlust for {}",
        "Epic {} adventures",
    ],
    "foodie": [
        "Feasting on {} dreams",
        "Culinary adventures with {}",
        "Taste buds dancing to {}",
        "Gourmet {} vibes",
        "Savoring every bite of {}",
        "Food coma from {}",
        "Epicurean delights: {}",
        "Nom nom {} heaven",
    ],
    "travel": [
        "Wanderlust: {}",
        "Exploring {} one destination at a time",
        "Journey to {}",
        "Passport stamped with {}",
        "Around the world in {} days",
        "Travel tales of {}",
        "Globe-trotting {}",
        "Adventure awaits in {}",
    ],
    "fitness": [
        "Sweat, smile, repeat: {}",
        "Fitness journey with {}",
        "Stronger every day: {}",
        "Beast mode: {}",
        "Gains and {} glory",
        "Workout warrior: {}",
        "Fit fam {} vibes",
        "Power through {}",
    ],
    "default": [
        "{}",
        "Just {}",
        "This moment: {}",
        "Simply {}",
        "All about {}",
        "Living for {}",
        "Embracing {}",
        "Celebrating {}",
        "In love with {}",
    ],
}

# Trending hashtags database 
TRENDING_HASHTAGS = [
    "#fyp", "#viral", "#explore", "#love", "#instagood", "#photooftheday",
    "#beautiful", "#happy", "#cute", "#fashion", "#followme", "#picoftheday",
    "#art", "#photography", "#nature", "#travel", "#fitness", "#beauty"
]

def load_custom_templates():
    """Load user-defined custom templates if available."""
    if os.path.exists("custom_templates.json"):
        with open("custom_templates.json", "r") as f:
            return json.load(f)
    return {}

def save_custom_template(style, template):
    """Save a custom template for a style."""
    custom = load_custom_templates()
    if style not in custom:
        custom[style] = []
    custom[style].append(template)
    with open("custom_templates.json", "w") as f:
        json.dump(custom, f)

def generate_captions(base: str, style="default", tone="casual", length="medium", emojis=True, include_trending=False):
    """
    Generate creative captions with enhanced options.

    Args:
        base: Base text for caption
        style: Caption style
        tone: Tone of caption
        length: Length preference
        emojis: Whether to include emojis
        include_trending: Whether to include trending hashtags

    Returns:
        List of generated captions
    """
    options = []
    custom_templates = load_custom_templates()

    # Combine built-in and custom templates
    all_templates = TEMPLATES.get(style, TEMPLATES["default"]) + custom_templates.get(style, [])

    if len(all_templates) < 5:
        all_templates.extend(TEMPLATES["default"][:5-len(all_templates)])

    for temp in all_templates[:5]:  # Generate up to 5 captions
        text = temp.format(base)
        text = apply_tone(text, tone)
        text = apply_length(text, length)
        if emojis:
            text = insert_emojis(text, tone)
        options.append(text)

    # Add trending hashtags if requested
    if include_trending:
        trending = random.sample(TRENDING_HASHTAGS, min(3, len(TRENDING_HASHTAGS)))
        for i, cap in enumerate(options):
            options[i] = cap + " " + " ".join(trending)

    return options
