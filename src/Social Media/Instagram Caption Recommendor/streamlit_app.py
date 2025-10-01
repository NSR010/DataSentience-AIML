import streamlit as st
import pandas as pd
from caption_generator import generate_captions, TEMPLATES, save_custom_template
from hashtag_utils import make_hashtags
from text_utils import translate_text
import json
import os

# Set page config with custom theme
st.set_page_config(
    page_title="InstaCaptioner Pro",
    layout="wide",
    page_icon="📸",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E1306C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .caption-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #E1306C;
    }
    .copy-btn {
        background-color: #E1306C;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for favorites
with st.sidebar:
    st.header("⭐ Favorites")
    if os.path.exists("favorites.json"):
        with open("favorites.json", "r") as f:
            favorites = json.load(f)
        for fav in favorites:
            st.text_area("", value=fav, height=50, key=fav[:20])
    else:
        st.write("No favorites yet!")

# Main content
st.markdown('<h1 class="main-header">📸 InstaCaptioner Pro</h1>', unsafe_allow_html=True)
st.markdown("### Generate creative Instagram captions with AI-powered suggestions!")

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "Enter keywords or short description:",
        height=120,
        placeholder="e.g., sunset at beach, coffee morning, workout motivation"
    )

with col2:
    style = st.selectbox("Style", ["witty", "inspirational", "aesthetic", "trending", "romantic", "motivational", "poetic", "sarcastic", "adventurous", "foodie", "travel", "fitness", "default"])
    tone = st.selectbox("Tone", ["casual", "funny", "formal", "inspirational", "sarcastic"])
    length = st.selectbox("Length", ["short", "medium", "long"])
    emojis = st.checkbox("Include emojis", value=True)
    trending_hashtags = st.checkbox("Include trending hashtags", value=False)
    language = st.selectbox("Translate to", ["none", "en", "es", "fr", "hi", "bn", "de", "it"])

    # Custom template section
    with st.expander("Add Custom Template"):
        custom_style = st.selectbox("Style for custom template", list(TEMPLATES.keys()))
        custom_template = st.text_input("Custom template (use {} for placeholder)", placeholder="e.g., Feeling {} today!")
        if st.button("Save Custom Template"):
            if custom_template.strip():
                save_custom_template(custom_style, custom_template)
                st.success("Custom template saved!")
            else:
                st.warning("Please enter a template.")

# Generate button
if st.button("🚀 Generate Captions", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("⚠️ Please enter some text or keywords first.")
    else:
        with st.spinner("Generating creative captions..."):
            captions = generate_captions(prompt, style=style, tone=tone, length=length, emojis=emojis)

        st.success(f"✅ Generated {len(captions)} caption options!")

        # Display captions
        for i, cap in enumerate(captions, 1):
            with st.container():
                st.markdown(f'<div class="caption-card">', unsafe_allow_html=True)
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**Option {i}**")
                    st.write(cap)
                    tags = make_hashtags(cap)
                    st.write("**Hashtags:** " + " ".join(tags))
                with col_b:
                    if st.button("📋 Copy", key=f"copy_{i}"):
                        st.session_state.clipboard = cap + "\n" + " ".join(tags)
                        st.success("Copied to clipboard!")
                    if st.button("⭐ Save", key=f"save_{i}"):
                        if not os.path.exists("favorites.json"):
                            with open("favorites.json", "w") as f:
                                json.dump([], f)
                        with open("favorites.json", "r") as f:
                            favs = json.load(f)
                        favs.append(cap)
                        with open("favorites.json", "w") as f:
                            json.dump(favs, f)
                        st.success("Saved to favorites!")
                st.markdown('</div>', unsafe_allow_html=True)

        # Translation section
        if language != "none":
            st.markdown("---")
            st.subheader("🌍 Translated Example")
            try:
                translated = translate_text(captions[0], language)
                st.write(f"**{language.upper()}:** {translated}")
            except Exception as e:
                st.warning(f"Translation failed: {str(e)}")

        # Export option
        st.markdown("---")
        if st.button("📄 Export All Captions"):
            export_data = "\n\n".join([f"Caption {i+1}: {cap}\nHashtags: {' '.join(make_hashtags(cap))}" for i, cap in enumerate(captions)])
            st.download_button(
                label="Download as TXT",
                data=export_data,
                file_name="instagram_captions.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for content creators | Keep creating amazing content!")
