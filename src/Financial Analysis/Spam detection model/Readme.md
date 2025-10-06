# SMS Spam Detection Using Deep Learning

This project is a text classification model that detects whether a given SMS/text message is Spam or Ham (Not Spam) using Deep Learning. The model was trained on a labeled SMS dataset and deployed with a user-friendly Streamlit web application for real-time predictions.

# Project Overview

Spam messages can be annoying and sometimes harmful. This project provides an automated solution to classify SMS messages as Spam or Ham with high accuracy using LSTM-based deep learning. Users can input any SMS and instantly get a prediction along with a confidence score.

# Dataset

Dataset Used: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Contains labeled SMS messages as spam or ham

Used for training and evaluating the deep learning model

# Project Workflow
1. Data Preprocessing

Cleaned and tokenized SMS messages

Converted text to sequences using Tokenizer

Applied padding to maintain equal input length (MAX_LEN = 100)

2. Model Architecture

Built using TensorFlow / Keras

Embedding Layer: Converts words into dense vectors

LSTM Layers: Capture sequential patterns in text

Dense Layers: Fully connected layers for classification

Sigmoid Output Layer: Produces binary output (Spam vs Ham)

3. Training & Evaluation

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Achieved high classification performance on the test data

4. Deployment

Model saved as spam_classifier.h5 / spam_classifier.keras

Tokenizer saved as tokenizer.pkl

Deployed using Streamlit for real-time predictions

# Features

✅ Detects spam messages with high accuracy
✅ Returns prediction confidence score
✅ Interactive web UI using Streamlit
✅ Supports any user-entered SMS/text message

# Example Predictions

Input: "Congratulations! You have won a $500 gift voucher. Click the link to claim."
Output: 🚨 Spam detected! (Confidence: 99%)

Input: "Hey, are we still meeting tomorrow at 5?"
Output: ✅ Ham (Not Spam) (Confidence: 100%)

# Technologies Used

Python

TensorFlow / Keras

NLTK / Text Preprocessing

Streamlit

Pickle 

# Project Structure
 SMS-Spam-Detection
 app.py                    
 spam_classifier.h5         
 tokenizer.pkl              
 README.md                  
 Text Classification for spam detection ml model.ipynb        

 # Author 
 👩‍💻 Smriti Pandey
📍 B.Tech CSE (AI & ML) | GTBIT

