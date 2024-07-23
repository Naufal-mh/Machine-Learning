import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import os

# Load the dataset
def load_dataset(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
    df = pd.read_csv(file_path)
    X = df[['r', 'g', 'b']].values
    y = df['color_name'].values
    return X, y

# Preprocess the dataset for CNN
def preprocess_data(X, y):
    X_processed = X.reshape(-1, 1, 1, 3) / 255.0  # Normalize the RGB values
    color_dict = {color: idx for idx, color in enumerate(np.unique(y))}
    y_processed = np.array([color_dict[color] for color in y])
    y_processed = tf.keras.utils.to_categorical(y_processed, num_classes=len(color_dict))
    return X_processed, y_processed, color_dict

# Build the CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset and build the model
file_path = 'colors.csv'
X, y = load_dataset(file_path)
X_processed, y_processed, color_dict = preprocess_data(X, y)
model = build_model((1, 1, 3), len(color_dict))
model.fit(X_processed, y_processed, epochs=100, verbose=0)

# Create a Streamlit app
st.title("Pendeteksi Warna")
st.write("Silakan ambil foto yang Anda inginkan untuk mendeteksi warna!")

# Create a camera feed component
camera_feed = st.camera_input("Kamera")

# Create a text component to display the color name and RGB value
color_text = st.text("Color: ")

# Create a button to start the color detection
start_button = st.button("Memulai Deteksi Warna")

if start_button:
    # Open the camera capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Get the center pixel
        center_pixel = resized_frame[240, 320]

        # Convert BGR to RGB
        rgb_value = resized_frame[::-1] / 255.0  # BGR to RGB and normalize
        rgb_value = rgb_value.reshape(1, 1, 1, 3)

        # Predict the color
        prediction = model.predict(rgb_value)
        color_name = color_dict[np.argmax(prediction)]

        # Display the color name and RGB value on the frame
        cv2.putText(resized_frame, f'Color: {color_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(resized_frame, f'RGB: {tuple(rgb_value[::-1])}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        image_placeholder.image(resized_frame, channels="RGB")

        # Update the camera feed component
        camera_feed.image(resized_frame)

        # Update the text component
        color_text.write(f"Color: {color_name}, RGB: {tuple(rgb_value[::-1])}")

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()