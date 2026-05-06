import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("best_mobilenetv2.keras")

# Class labels
class_names = [
    "Abnormal Heartbeat",
    "Covid-19",
    "MI",
    "MI History",
    "Normal"
]

st.title("ECG Classification")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded ECG", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
