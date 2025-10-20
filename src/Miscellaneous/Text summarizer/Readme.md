# AI Text Summarizer Web App
An intelligent text summarization web application built with Hugging Face Transformers and Streamlit.

This app allows users to summarize long articles or news content into concise 3-line summaries - either by pasting text or entering a URL.

# 🚀 Features

 🧠 Uses Hugging Face's BART-large-CNN model for high-quality abstractive summarization.

🌐 Supports URL-based content extraction (fetches full article text automatically).

🧾 3-line summary generation for clear and precise understanding.

🖥️ Built with a clean, interactive Streamlit web interface.

⚡ Handles long texts by splitting content into manageable chunks.

# Tech Stack

Python 3.10+

Streamlit

Transformers (Hugging Face)

Requests

BeautifulSoup4

Regex (re)

# Installation

Clone this repository

https://github.com/student-smritipandey/AI-Text-Summarizer-Web-App.git

cd Text summarizer

Create and activate a virtual environment

python -m venv venv

venv\Scripts\activate # For Windows

source venv/bin/activate # For Mac/Linux

Install dependencies

pip install -r requirements.txt

# Run the App

Start the Streamlit server:

streamlit run app.py

Then open the app in your browser (default: http://localhost:8501).

# Usage

Select your input type - Text or URL.

Paste your article or enter a news article URL.

Click "Summarize".

Get your 3-line AI-generated summary instantly!

# Project Structure

AI-Text-Summarizer/

 app.py # Streamlit web application

 requirements.txt # Python dependencies

 README.md # Project documentation

Text summarizer.ipynb 



# Requirements

Create a requirements.txt file with:

streamlit

transformers

torch

requests

beautifulsoup4

lxml_html_clean

# Author

Smriti Pandey 🎓 B.Tech CSE (AIML) | Guru Tegh Bahadur Institute of Technology

💼 Passionate about Deep Learning, NLP, and AI-driven Applications