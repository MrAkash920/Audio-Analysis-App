import streamlit as st
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
#import soundfile as sf
import os
import librosa
import numpy as np
import pandas as pd

# Load pre-trained models
emotion_classifier = pipeline("text-classification", model="bert-base-cased",
                              tokenizer="bert-base-cased")

model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
audio_model = Wav2Vec2ForCTC.from_pretrained(model_name)


def extract_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Extract pitch
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    mean_pitch = np.mean(pitch)

    return mfccs, mean_pitch




st.title("Audio Analysis App")
st.write("Upload an audio file to analyze emotion and aggression!")

# Audio file upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Process the uploaded audio file
    audio_input, sample_rate = torchaudio.load(uploaded_file)

    # Transcribe audio
    inputs = tokenizer(audio_input.squeeze().numpy(), return_tensors="pt", padding="longest")
    transcription = audio_model(input_values=inputs.input_values).logits.argmax(dim=-1)
    transcription_text = tokenizer.batch_decode(transcription)

    # Classify the transcribed text for emotion and aggression
    emotion_result = emotion_classifier(transcription_text[0])
    emotion_label = "happy" if emotion_result[0]['label'] == 'LABEL_0' else "sad"

    # Create dataframes for tables
    emotion_data = {"Audio File Name": [uploaded_file.name], "Predicted Emotion": [emotion_label]}
    aggression_data = {"Audio File Name": [uploaded_file.name], "Predicted Aggression": ["Not Implemented"]}

    # Create line charts
    emotion_chart_data = pd.DataFrame({"Time": [0], "Emotion": [emotion_label]})
    aggression_chart_data = pd.DataFrame({"Time": [0], "Aggression": ["Not Implemented"]})

    # Display results
    st.audio(uploaded_file)
    st.write("Transcription:")
    st.write(transcription_text[0])

    st.subheader("Emotion Prediction")
    st.write("Predicted Emotion: ", emotion_label)
    st.table(pd.DataFrame(emotion_data))

    st.subheader("Emotion Prediction Chart")
    st.line_chart(emotion_chart_data.set_index("Time"))
