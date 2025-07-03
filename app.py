import streamlit as st
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import zipfile

@st.cache_data
def load_data():
    with zipfile.ZipFile("TXT_GEN.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    df = pd.read_csv("TXT_GEN.csv")
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    df.dropna(subset=['text', 'subject'], inplace=True)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['subject'])
    return df, label_encoder

@st.cache_resource
def load_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, gpt2_model, bert

@st.cache_resource
def train_classifiers(X_train_embed, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000).fit(X_train_embed, y_train),
        "Random Forest": RandomForestClassifier().fit(X_train_embed, y_train),
        "SVM": SVC().fit(X_train_embed, y_train),
        "Naive Bayes": GaussianNB().fit(X_train_embed, y_train),
        "Gradient Boosting": GradientBoostingClassifier().fit(X_train_embed, y_train)
    }
    return models

st.title("Text Generation + Classification (Transformer + ML)")

df, label_encoder = load_data()
tokenizer, gpt2_model, bert = load_models()

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
X_train_embed = bert.encode(X_train.tolist())
X_test_embed = bert.encode(X_test.tolist())

models = train_classifiers(X_train_embed, y_train)

user_input = st.text_area("Enter a text prompt for GPT-2:", value=X_train.iloc[0][:100])

if st.button("Generate & Classify"):
    with st.spinner("Generating text..."):
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        output = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        st.success("Generated Text:")
        st.write(generated)

        emb = bert.encode([generated])
        st.subheader("Classification Results:")
        for name, model in models.items():
            pred = model.predict(emb)[0]
            st.write(f"**{name}**: {label_encoder.inverse_transform([pred])[0]}")
