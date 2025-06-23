# Employee Mood Detection System

This project leverages **machine learning and deep learning** to detect employee moods using **text inputs** and **facial expressions**, enabling proactive support and task alignment based on emotional states.

---

## Features

### 1. Text-Based Emotion Detection
- Takes user input (like a message or comment).
- Predicts emotion using a trained machine learning model (e.g., SVM).
- Returns friendly suggestions depending on the detected mood.
- Trained on a labeled dataset with emotions like: `joy`, `sadness`, `anger`, `fear`, `love`, and `surprise`.

### 2. Facial Emotion Detection
- Accepts image input of a face (48x48 grayscale).
- Uses a CNN (Convolutional Neural Network) trained on the **FER2013 dataset**.
- Detects facial emotions: `Happy`, `Sad`, `Angry`, `Neutral`, `Fear`, `Surprise`, `Disgust`.
- Gives personalized suggestions based on facial emotion too.

---

## 📁 Project Structure

employee-mood-detector/
├── text_model.pkl # Trained ML model for text emotion
├── vectorizer.pkl # TfidfVectorizer for text preprocessing
├── emotion_face_model.h5 # CNN model for facial emotion detection
├── collab_1_text_model.ipynb # Training notebook for text model
├── collab_2_face_model.ipynb # Training notebook for face model
├── collab_3_combiner.ipynb # Combined inference logic
└── README.md

## Tech Stack

- **Python**
- **Scikit-learn**: for text-based model
- **TensorFlow / Keras**: for CNN facial recognition model
- **Pandas & NumPy**: for data handling
- **OpenCV / PIL**: for image preprocessing
- **Colab**: for training and prototyping

---

## Use Case

This system is designed for organizations to:

- Detect employees facing **stress**, **burnout**, or **negative emotions**.
- Offer task suggestions based on mood to boost productivity.
- Notify HR or managers for intervention if continuous negative emotion is detected.

---

## Future Scope

- Add **Streamlit web interface** for easier real-time access.
- Integrate **voice tone analysis**.
- Deploy as an internal HR dashboard.
- Add **chatbot integration** for conversational emotion check-ins.

---

## Note

This project consists of 3 Google Colab notebooks:
1. Training the **text-based emotion detection model**.
2. Training the **facial emotion recognition model**.
3. Combining both into a **unified emotion detection system**.

