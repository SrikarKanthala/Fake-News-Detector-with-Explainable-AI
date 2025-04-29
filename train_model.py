
import pandas as pd
import xgboost as xgb
from transformers import BertTokenizer, BertModel
import torch

# Load dataset (simplified)
df = pd.read_csv('news_sample.csv')
df = df.dropna()
df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})

# Load BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

# Generate embeddings
X = []
for text in df['text']:
    emb = get_bert_embedding(text)
    X.append(emb)

X = pd.DataFrame(X)
y = df['label']

# Train XGBoost model
dtrain = xgb.DMatrix(X, label=y)
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
model = xgb.train(params, dtrain, num_boost_round=10)
model.save_model('xgb_model.json')
