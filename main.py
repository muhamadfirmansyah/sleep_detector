import streamlit as st
import joblib
import os
import librosa
import numpy as np

# Load the saved model and scaler
MODEL_PATH = "snoring_rf_model.joblib"
SCALER_PATH = "scaler.joblib"

rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Function to extract features from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(zcr), np.std(zcr),
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
    ])

# Function to predict snoring or non-snoring
def predict_snoring(file_path):
    try:
        features = extract_features(file_path)
        features = scaler.transform([features])  # Standardize features
        probabilities = rf_model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)  # Get class with highest probability
        confidence = probabilities[prediction]  # Confidence of prediction
        label = "Snoring" if prediction == 1 else "Not Snoring"
        return label, confidence
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return None, None

# Streamlit UI setup
st.title("Snoring Detection App")
st.markdown("Upload an audio file to detect if it's snoring or not.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict and display results
    label, confidence = predict_snoring(temp_file_path)
    if label:
        st.audio(temp_file_path, format="audio/wav")
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")
    os.remove(temp_file_path)  # Remove the temporary file after prediction
