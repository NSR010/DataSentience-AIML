import streamlit as st
import re
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# ---------------------------
# Helper Functions
# ---------------------------

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def chunk_text(text, max_chars=1000):
   
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind('.', 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at+1])
        text = text[split_at+1:]
    if text:
        chunks.append(text)
    return chunks

def simple_sent_tokenize(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s]

def to_three_lines(summary_text):
    summary_text = clean_text(summary_text)
    sents = simple_sent_tokenize(summary_text)
    if len(sents) >= 3:
        return "\n".join(sents[:3])
    words = summary_text.split()
    n = len(words)
    sizes = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    lines, idx = [], 0
    for sz in sizes:
        lines.append(' '.join(words[idx:idx+sz]))
        idx += sz
    return "\n".join(lines)

# ---------------------------
# Hugging Face Summarizer
# ---------------------------

hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def summarize_with_hf(text):
    text = clean_text(text)
    if len(text) < 50:
        return "Text too short to summarize."

    chunks = chunk_text(text, max_chars=1000)
    summaries = []

    for chunk in chunks:
        if len(chunk.strip()) < 50:
            continue
        try:
            result = hf_summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(result[0]['summary_text'])
        except Exception as e:
            print(f"Skipped chunk due to error: {e}")

    if not summaries:
        return "Could not generate summary. Text may be too short or fragmented."

    return to_three_lines(" ".join(summaries))

# ---------------------------
# Fetch article from URL
# ---------------------------

def fetch_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    text = ' '.join(paragraphs) if paragraphs else soup.get_text()
    title = soup.title.string if soup.title else "Untitled"
    return title, text

# ---------------------------
# Streamlit App
# ---------------------------

st.title("📝 AI Text Summarizer")

input_type = st.radio("Select input type:", ["Text", "URL"])

text = ""
title = "Manual Input"

if input_type == "Text":
    text = st.text_area("Paste your article/text here:", height=300)
elif input_type == "URL":
    url = st.text_input("Enter article URL:")
    if url:
        try:
            title, text = fetch_article_text(url)
        except Exception as e:
            st.error(f"Failed to fetch URL: {e}")

if st.button("Summarize"):
    if text and len(text.strip()) > 50:
        with st.spinner("Generating summary..."):
            summary = summarize_with_hf(text)
        st.subheader(f"Title: {title}")
        st.text_area("3-Line Summary:", summary, height=150)
    else:
        st.warning("Please provide sufficient text to summarize (50+ characters).")
