"""
DO NOT TRY IT, IT'S HORRIBLE
That was a failed attempt for another method to 
detect the lip Reading by recording then predicting
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

# Model definition
vocab = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(vocab), oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

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

def process_and_predict(video_path, sequence_length=10):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    predictions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if unable to capture a frame

        if len(frame_sequence) < sequence_length:
            frame_sequence.append(frame)
        else:
            # Preprocess and predict the collected frame sequence
            prediction_text = process_frame_sequence(frame_sequence)
            predictions.append(prediction_text)
            frame_sequence = []

    cap.release()
    return predictions

def process_frame_sequence(frame_sequence):
    # Convert frames to grayscale, resize, normalize, and predict
    processed_frames = np.array([tf.image.resize(tf.image.rgb_to_grayscale(f), [46, 140]) / 255.0 for f in frame_sequence])
    processed_frames = np.expand_dims(processed_frames, axis=0)  # Add batch dimension
    predictions = model.predict(processed_frames)
    decoded_predictions = tf.keras.backend.ctc_decode(predictions, input_length=[predictions.shape[1]])[0][0].numpy()
    prediction_text = tf.strings.reduce_join(num_to_char(decoded_predictions)).numpy().decode('utf-8')
    return prediction_text

# Record video from webcam
def record_video(filename='recorded_video.avi', duration=10):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    start_time = cv2.getTickCount()
    while int((cv2.getTickCount() - start_time)/cv2.getTickFrequency()) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return filename

# Main script
if __name__ == "__main__":
    video_file = record_video(duration=3)  # Record for 5 seconds
    predictions = process_and_predict(video_file)
    print("Predictions:", predictions)
