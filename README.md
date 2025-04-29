
# Fake News Detector with Explainable AI

This project uses a hybrid approach to detect fake news articles by combining BERT embeddings with an XGBoost classifier.
SHAP is used to explain the model predictions, and the entire tool is deployed via Streamlit for real-time article testing.

## Features
- BERT-based text embeddings
- XGBoost classification
- SHAP explainability
- Streamlit web app

## Dataset
We use a cleaned fake news dataset adapted from Kaggle: [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Launch app: `streamlit run app.py`
