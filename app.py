import streamlit as st
import pandas as pd
import zipfile
import os
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.set_page_config(page_title="Text Generation and Classification", layout="wide")
st.title("Text Generation and Classification (Transformer + ML)")

# --- Load Dataset from ZIP ---
@st.cache_data(show_spinner=False)
def load_data_from_zip():
    zip_path = "TXT_GEN.zip"
    csv_filename = "TXT_GEN.csv"

    if not os.path.exists(zip_path):
        st.error(f"{zip_path} not found. Please upload it to the root of the repo.")
        st.stop()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall()
    except zipfile.BadZipFile:
        st.error("Invalid ZIP file format.")
        st.stop()

    if not os.path.exists(csv_filename):
        st.error("TXT_GEN.csv not found after extracting ZIP.")
        st.stop()

    df = pd.read_csv(csv_filename)
    df['text'] = df['text'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    df.dropna(subset=['text', 'subject'], inplace=True)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['subject'])
    return df, label_encoder

# --- Load Models ---
@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, gpt2_model, bert

# --- Train Classifiers ---
def train_classifiers(X_train_embed, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000).fit(X_train_embed, y_train),
        "Random Forest": RandomForestClassifier().fit(X_train_embed, y_train),
        "SVM": SVC().fit(X_train_embed, y_train),
        "Naive Bayes": GaussianNB().fit(X_train_embed, y_train),
        "Gradient Boosting": GradientBoostingClassifier().fit(X_train_embed, y_train)
    }
    return models

# Load data and models
with st.spinner("Loading dataset and models..."):
    df, label_encoder = load_data_from_zip()
    tokenizer, gpt2_model, bert = load_models()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Prompt input
st.subheader("Enter a text prompt for GPT-2:")
user_input = st.text_area("Prompt:", value=X_train.iloc[0][:100], height=150)

# Placeholder for trained models
trained_models = None
X_train_embed, X_test_embed = None, None

if st.button("Generate Text and Train Classifiers"):
    if user_input.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating text with GPT-2..."):
            inputs = tokenizer.encode(user_input, return_tensors="pt")
            outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Text Generated:")
            st.write(generated_text)

        with st.spinner("Generating BERT embeddings..."):
            X_train_embed = bert.encode(X_train.tolist(), show_progress_bar=True)
            X_test_embed = bert.encode(X_test.tolist(), show_progress_bar=True)
            generated_embed = bert.encode([generated_text])

        with st.spinner("Training classifiers..."):
            trained_models = train_classifiers(X_train_embed, y_train)
            st.success("Models trained.")

        st.subheader("Classification Results for Generated Text")
        for name, model in trained_models.items():
            pred = model.predict(generated_embed)[0]
            st.write(f"{name}: {label_encoder.inverse_transform([pred])[0]}")
