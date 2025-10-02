# InstaCaptioner Pro

A powerful and user-friendly tool to generate creative Instagram captions with AI-powered suggestions. Create engaging captions tailored to your style, tone, and preferences with just a few clicks!

## Features

- **Multiple Caption Styles**: Choose from 13+ styles including witty, inspirational, aesthetic, trending, romantic, motivational, poetic, sarcastic, adventurous, foodie, travel, fitness, and more.
- **Tone Customization**: Adjust the tone to casual, funny, formal, inspirational, or sarcastic.
- **Length Control**: Generate short, medium, or long captions.
- **Emoji Integration**: Automatically add relevant emojis based on tone and sentiment.
- **Trending Hashtags**: Include popular trending hashtags to boost visibility.
- **Hashtag Generation**: Automatically generate relevant hashtags from your caption.
- **Multi-language Support**: Translate captions to English, Spanish, French, Hindi, Bengali, German, Italian, and more.
- **Custom Templates**: Add your own custom caption templates for personalized results.
- **Favorites System**: Save your favorite captions for future use.
- **Export Functionality**: Download all generated captions as a text file.
- **Responsive UI**: Built with Streamlit for a smooth, interactive experience.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/instagram-caption-recommender.git
   cd instagram-caption-recommender
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to start using the app.

## Usage

1. **Enter Keywords**: Type in keywords or a short description of your post (e.g., "sunset at beach", "coffee morning", "workout motivation").

2. **Customize Options**:
   - Select a caption style from the dropdown.
   - Choose the desired tone.
   - Set the preferred length.
   - Toggle emojis and trending hashtags as needed.
   - Select a language for translation (optional).

3. **Generate Captions**: Click the "🚀 Generate Captions" button to create up to 5 caption options.

4. **Review and Use**:
   - Browse through the generated captions.
   - Copy captions to clipboard or save to favorites.
   - View automatically generated hashtags.
   - Translate captions if desired.
   - Export all captions for offline use.

5. **Advanced Features**:
   - Add custom templates in the sidebar for personalized caption styles.
   - Access saved favorites in the sidebar.

## Requirements

- streamlit
- pillow
- googletrans==4.0.0rc1
- langdetect

## Project Structure

```
instagram-caption-recommender/
│
├── caption_generator.py      # Core caption generation logic and templates
├── streamlit_app.py          # Main Streamlit application UI
├── text_utils.py             # Text processing utilities (tone, length, emojis, translation)
├── hashtag_utils.py          # Hashtag generation functionality
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── custom_templates.json     # User-defined custom templates (created automatically)
└── favorites.json            # Saved favorite captions (created automatically)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Disclaimer

This tool uses Google Translate for language translation features. Please be aware of Google's terms of service when using translation features extensively.

---

Made with ❤️ for content creators. Keep creating amazing content!

Author - Anu