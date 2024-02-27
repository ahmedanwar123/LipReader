import cv2
import numpy as np
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt
import imageio
import os
import gdown
from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

# Define char_to_num and num_to_char before using them
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(1, 140, 46, 128), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

model.summary()

def process_frame(frame):
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.resize(frame, [140, 46])
    frame = frame[190:236, 80:220, :]

    # Normalize the frame
    mean = tf.math.reduce_mean(frame)
    std = tf.math.reduce_std(tf.cast(frame, tf.float32))
    normalized_frame = tf.cast((frame - mean), tf.float32) / std

    # Expand dimensions to match the model input shape
    normalized_frame = tf.expand_dims(normalized_frame, axis=0)
    normalized_frame = tf.expand_dims(normalized_frame, axis=-1)  # Add the channel dimension

    # Pad the frame to match the expected shape
    pad_size = [(0, 0), (0, 1), (0, 0), (0, 1)]  # Add padding to the second and last dimensions
    normalized_frame = tf.pad(normalized_frame, pad_size)

    # Send the frame to the model for processing
    predictions = model.predict(normalized_frame)

    # Decode the predictions
    decoded_predictions = tf.keras.backend.ctc_decode(predictions, input_length=[75], greedy=True)[0][0].numpy()

    # Display the frame with the predictions
    cv2.putText(frame, tf.strings.reduce_join([num_to_char(word) for word in decoded_predictions]).numpy().decode('utf-8'),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify video file path

while True:
    # _, frame = cap.read()
    ret, frame = cap.read()

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Process the frame using the model
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Video', processed_frame)

cap.release()
cv2.destroyAllWindows()
