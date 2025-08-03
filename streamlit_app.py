import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and tools
model = joblib.load('complaint_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("📨 Complaint Classifier")
st.write("Paste a customer complaint below and the model will classify it.")

user_input = st.text_area("Enter complaint text:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        category = label_encoder.inverse_transform([pred])[0]
        st.success(f"**Predicted Category:** {category}")
