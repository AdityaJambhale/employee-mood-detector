
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# Load models
text_model = joblib.load("text_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
face_model = tf.keras.models.load_model("emotion_face_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_suggestions = {
    "Angry": "😡 Try a quick walk to cool off 🚶",
    "Disgust": "😖 Step away and reset your mind 🧘",
    "Fear": "😨 Deep breaths help! You're safe here 💙",
    "Happy": "😊 Perfect time to take on focused tasks 💪",
    "Sad": "😔 Take a short break, listen to music 🎧",
    "Surprise": "😲 Channel that energy into something creative 🎨",
    "Neutral": "😐 Maintain the calm and keep going 🔄"
}

st.title("🧠 Mood Analyzer — Text + Face")
st.write("Enter your feeling in words and upload a selfie to detect your emotion.")

text_input = st.text_input("💬 How are you feeling today?")
uploaded_image = st.file_uploader("📸 Upload a face image", type=["jpg", "jpeg", "png"])

if st.button("🔍 Analyze") and text_input and uploaded_image:
    text_vec = vectorizer.transform([text_input])
    text_pred = text_model.predict(text_vec)[0]

    img = Image.open(uploaded_image).convert("RGB")
    img_resized = img.resize((48, 48))
    img_array = keras_image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    face_pred_idx = tf.argmax(face_model.predict(img_array)[0]).numpy()
    face_pred = emotion_labels[face_pred_idx]

    final_emotion = text_pred if text_pred == face_pred else text_pred
    suggestion = emotion_suggestions.get(final_emotion, "🙂 Stay balanced and do your thing!")

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.subheader(f"🧠 Detected Emotion: {final_emotion}")
    st.success(f"💡 Suggestion: {suggestion}")
