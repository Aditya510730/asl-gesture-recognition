# ASL Gesture Recognition

A real-time American Sign Language (ASL) hand gesture recognition system built with MediaPipe and LSTM neural networks. The system uses a standard webcam to detect hand landmarks and classify ASL letters A through Z in real time with no specialized hardware required.

## Demo

The system captures live video, extracts hand keypoints using MediaPipe, and passes sequences of frames through a trained LSTM network to predict ASL letters in real time. Recognized letters are displayed on screen along with confidence scores.

## Motivation

This project was built to make technology more accessible for the Deaf community. Traditional sign language recognition systems often require expensive hardware or controlled environments. This system works with any standard webcam, making it practical for everyday use.

## How It Works

1. A webcam captures live video frames
2. MediaPipe extracts 21 hand landmarks (x, y, z coordinates) per frame
3. Keypoints from 14 consecutive frames are compiled into a sequence
4. The LSTM model classifies the sequence into one of 26 ASL letter categories
5. Predictions above a confidence threshold of 0.8 are displayed on screen in real time

## Model Architecture

- Input Layer: sequences of flattened hand keypoints (14 frames x 63 keypoints)
- Multiple LSTM layers to capture temporal dependencies across frames
- Dense fully connected layer for classification
- Output Layer: softmax probabilities over 26 gesture classes (A through Z)

The model was trained on custom data collected via webcam using OpenCV, with images stored per gesture category.

## Project Structure

- data.py — Collects training data by saving hand keypoint sequences from webcam input
- trainmodel.py — Trains the LSTM model on the collected keypoint sequences
- app.py — Runs real-time inference using the trained model and webcam feed
- model.h5 — Pre-trained LSTM model weights
- MP_Data — Keypoint sequences used for training

## Requirements

- Python 3.x
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy

Install dependencies:
pip install tensorflow mediapipe opencv-python numpy

## Usage

Step 1 — Collect training data:
python data.py

Step 2 — Train the model:
python trainmodel.py

Step 3 — Run real-time ASL recognition:
python app.py

Press q to quit the webcam feed.

## Research Paper

This project is accompanied by a technical paper titled Real-Time Hand Gesture Recognition Using MediaPipe and LSTM, which documents the full methodology including data collection, feature extraction, model training, and real-time inference results.

## References

- Google MediaPipe Hands Documentation
- Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

## Author

Aditya Panday
De Anza College, Cupertino CA
pandey.aditya28aug@gmail.com
