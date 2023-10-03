# Audio Analysis App

## Overview

The **Audio Analysis App** is a Python application built with Streamlit that allows users to upload audio files (in WAV or MP3 format) for analysis. The analysis includes audio transcription, emotion prediction, and presentation of the results in a user-friendly interface.

## Demo
Demo Link [https://audioanalysisapp.streamlit.app/]

## Features

- **Audio Transcription**: Transcribes the uploaded audio into text using the pre-trained Facebook Wav2Vec2 model.
- **Emotion Prediction**: Predicts the emotion conveyed in the transcribed text using the BERT-based text classification model.
- **User-Friendly Interface**: Streamlit provides an intuitive and interactive user interface for uploading and analyzing audio files.
- **Visualization**: Presents the transcribed text and predicted emotion in tables and charts for easy interpretation.

## How to Use

1. Clone the repository:

   ```shell
   git clone https://github.com/MrAkash920/Audio-Analysis-App.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Run the app:

   ```shell
   streamlit run app.py
   ```

4. Open your web browser and navigate to the local address provided by Streamlit.

5. Upload an audio file (in WAV or MP3 format) for analysis.

6. View the transcription, predicted emotion, and emotion prediction chart.

## Pre-trained Models

The app leverages the following pre-trained models:

- **Wav2Vec2**: The Facebook Wav2Vec2 model is used for audio transcription.
- **BERT-based Emotion Classifier**: A BERT-based text classification model is used for emotion prediction from the transcribed text.

## Outcome

![Outcome](https://github.com/MrAkash920/Audio-Analysis-App/raw/main/Outcome.png)

## Future Improvements

- **Improved Emotion Classification**: Enhance the emotion classification model to provide more nuanced emotions.
- **Real-time Emotion Analysis**: Implement real-time emotion analysis and visualization as the audio plays.
- **Confidence Scores**: Display confidence scores for emotion predictions.
- **Aggression Prediction**: Implement a model or feature for predicting aggression in the transcribed text.
- **Audio Feature Visualization**: Visualize audio features such as MFCCs and pitch.
- **Error Handling**: Implement error handling and user feedback for cases where audio processing fails.

## Contribution

Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

