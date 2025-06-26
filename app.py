#========================import packages=========================================================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

#========================loading the save files==================================================
lg = pickle.load(open('logistic_regresion.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))

# =========================repeating the same functions==========================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(lg.predict(input_vectorized))

    return predicted_emotion,label



#==================================creating app====================================
# Add custom CSS for a more attractive UI
st.markdown('''
    <style>
    .main-title {
        color: #2E86AB;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .subtitle {
        color: #666;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .emotion-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 1.2rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.07);
    }
    .emotion-label {
        font-size: 2rem;
        font-weight: bold;
        color: #28a745;
        margin-bottom: 0.5rem;
    }
    .prob-label {
        font-size: 1.1rem;
        color: #6c757d;
    }
    .stTextInput > div > input {
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #b2bec3;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        font-size: 1.1rem;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        margin-top: 0.5rem;
    }
    </style>
''', unsafe_allow_html=True)

st.markdown('<div class="main-title">Six Human Emotions Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect Joy, Fear, Anger, Love, Sadness, or Surprise from your text</div>', unsafe_allow_html=True)
st.write("<hr>", unsafe_allow_html=True)

# taking input from user
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    predicted_emotion, label = predict_emotion(user_input)
    st.markdown(f'''<div class="emotion-box">
        <div class="emotion-label">{predicted_emotion}</div>
        <div class="prob-label">Probability: {label}</div>
    </div>''', unsafe_allow_html=True)