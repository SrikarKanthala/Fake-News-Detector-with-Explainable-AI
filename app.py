
import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load trained XGBoost model
model = xgb.Booster()
model.load_model('xgb_model.json')

# Function to convert text to BERT embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Load SHAP explainer
explainer = shap.TreeExplainer(model)

# Streamlit UI
st.title("📰 Fake News Detector with Explainable AI")
input_text = st.text_area("Enter a news article:")

if st.button("Analyze"):
    if input_text:
        emb = get_bert_embedding(input_text)
        dmatrix = xgb.DMatrix(emb)
        prediction = model.predict(dmatrix)[0]
        label = "Fake" if prediction > 0.5 else "Real"
        st.subheader(f"Prediction: {label} ({prediction:.2f})")
        
        shap_values = explainer.shap_values(emb)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, emb, show=False)
        st.pyplot()
    else:
        st.warning("Please enter some text.")
