import cv2
import numpy as np
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt
import imageio
import os
import gdown
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

# Define vocabulary and lookup layers
vocab = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(vocab), oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Model definition 
model = Sequential([
    Conv3D(128, kernel_size=(3, 3, 3), activation='relu', input_shape=(10, 46, 140, 1), padding='same'),
    MaxPool3D(pool_size=(1, 2, 2)),
    Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPool3D(pool_size=(1, 2, 2)),
    Conv3D(75, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPool3D(pool_size=(1, 2, 2)),
    TimeDistributed(Flatten()),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Dense(len(char_to_num.get_vocabulary())+1, activation='softmax')
])

model.summary()

def process_and_predict(frame_sequence):
    # Preprocess the frames: Convert to grayscale, resize, normalize
    processed_frames = np.array([tf.image.resize(tf.image.rgb_to_grayscale(f), [46, 140]) / 255.0 for f in frame_sequence])
    
    # Batch dimension
    processed_frames = np.expand_dims(processed_frames, axis=0)  

    # Prediction
    predictions = model.predict(processed_frames)

    # Decoding predictions
    decoded_predictions = tf.keras.backend.ctc_decode(predictions, input_length=[predictions.shape[1]])[0][0].numpy()

    # Convert numerical predictions to text using the num_to_char layer
    prediction_text = tf.strings.reduce_join(num_to_char(decoded_predictions)).numpy().decode('utf-8')

    return prediction_text

# Parameters for frame sequence
sequence_length = 10  # Number of frames in a sequence (using gpu we can get more)
frame_sequence = []  # Store frames

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if unable to capture a frame

    # R eal-time visualization
    cv2.imshow('Live LipRead', frame)

    # Append the captured frame to our sequence
    if len(frame_sequence) < sequence_length:
        frame_sequence.append(frame)
    else:
        # Process and predict the collected frame sequence
        prediction_text = process_and_predict(frame_sequence)

        # Display the prediction result
        print("Predicted Text:", prediction_text)

        # Clear frame sequence to start collecting a new sequence
        frame_sequence = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()