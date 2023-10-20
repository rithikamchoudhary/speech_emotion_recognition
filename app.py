import joblib
import librosa
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__) # Initialising Flask Application

@app.route("/")
def index():
    return render_template("index.html")

# Define a function to extract features from an audio file
def extract_feature(file_name, mfcc, chroma, mel):
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    except Exception as e:
        return None
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

# Define a function to predict the emotion from an audio file
def predict_emotion(model, file_path):
    feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    if feature is None:
        return None
    feature = feature.reshape(1, -1)
    try:
        emotion = model.predict(feature)[0]
    except Exception as e:
        return None
    return emotion

@app.route("/predict", methods=["POST"])
def predict():
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return "No audio file provided."
    if audio_file.filename == "":
        return "No audio file selected."
    if audio_file:
        file_ext = audio_file.filename.split(".")[-1]
        if file_ext not in ["wav", "mp3"]:
            return "Unsupported file type. Please upload a WAV or MP3 file."
        if audio_file.content_length > 50 * 1024 * 1024:
            return "File too large. Please upload a file under 50 MB."
        model = joblib.load('best_model.joblib')
        emotion = predict_emotion(model, audio_file)
        if emotion is None:
            return "Error processing audio file. Please try a different file."
        result = f"Predicted Emotion : {emotion}"
        return result
    return "Error processing audio file. Please try a different file."

if __name__ == "__main__":
    app.run(debug=True)