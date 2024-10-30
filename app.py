import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.markdown("""
    <style>
    /* Background with solid color and gradient overlay */
    body {
        background: #282c34; /* Solid color background */
        background: linear-gradient(135deg, rgba(255, 154, 158, 0.8), rgba(250, 208, 196, 0.8)), #282c34;
        font-family: Arial, sans-serif;
        color: #f3f3f3;
    }
    
    /* Centered title with shadow */
    .title {
        text-align: center;
        font-size: 3em;
        color: #282c34;
        font-weight: bold;
        padding: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Styling the text area */
    .stTextArea {
        background-color: #fff;
        border-radius: 10px;
        border: 2px solid #007bff;
        padding: 8px;
        color: #333;
        font-size: 1.1em;
    }
    
    /* Button styling with shadow and gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.2em;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    
    /* Result header styling with smooth color transition */
    .result-header {
        text-align: center;
        font-size: 2.8em;
        color: #ffffff;
        font-weight: bold;
        padding-top: 20px;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Styling for Spam result */
    .spam {
        color: #ff4d4d;
        animation: fadeIn 1s;
    }
    /* Styling for Not Spam result */
    .not-spam {
        color: #28d79f;
        animation: fadeIn 1s;
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌈 Email/SMS Spam Classifier</div>', unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter the message", placeholder="Type your message here...", max_chars=500)

if st.button('🔍 Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.markdown('<div class="result-header spam">🚫 Spam</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-header not-spam">✅ Not Spam</div>', unsafe_allow_html=True)
