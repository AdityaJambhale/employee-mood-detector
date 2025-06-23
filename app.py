import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 😄 Mood-based suggestions
emotion_suggestions = {
    "sadness": "Take a short break, listen to music 🎧",
    "anger": "Try a quick walk to cool off 🚶",
    "love": "Spread that love – maybe write a kind message 💌",
    "joy": "Perfect time to take on focused tasks 💪",
    "fear": "Do some deep breathing – you're safe 🧘",
    "surprise": "Enjoy the moment and stay curious 🤯",
    "neutral": "You're steady – stay on track ✅"
}

# 🚀 Load your trained model and vectorizer
with open("text_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# 🎨 Streamlit UI
st.title("🧠 Employee Mood Detector (Text-Based)")
st.write("Enter a sentence and get an emotion prediction with a friendly suggestion!")

user_input = st.text_input("💬 How are you feeling today?", "")

if st.button("Detect Mood"):
    if user_input.strip() == "":
        st.warning("Please enter something to analyze!")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        st.success(f"🧠 Detected Emotion: **{prediction.capitalize()}**")
        st.info(f"💡 Suggestion: {emotion_suggestions.get(prediction, '🙂 Stay balanced!')}")
