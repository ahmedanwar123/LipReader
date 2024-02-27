# Import necessary libraries
import streamlit as st
import os
import imageio
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipReader')
    st.info('This application is developed from the LipReading deep learning model.')

st.title('LipNet Full Stack App')
# Generating a list of options or videos
options = os.listdir(os.path.join('/media/ahmed/Metropolis/LipReader', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('/media/ahmed/Metropolis/LipReader', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # Set environment variable to use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Save frames using imageio
        frames = []
        for frame in video:
            # Convert EagerTensor to NumPy array and convert to RGB format
            frame_np = np.array(frame).astype(np.uint8)
            
            # Convert frame to RGB
            frame_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(frame_np))
            
            # Convert frame to bytes
            img_bytes = BytesIO()
            Image.fromarray(frame_rgb.numpy()).save(img_bytes, format='JPEG')
            frames.append(Image.open(BytesIO(img_bytes.getvalue())))

        # Save frames as gif
        gif_path = 'animation.gif'
        imageio.mimsave(gif_path, frames, fps=10)
        st.image(gif_path, width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
