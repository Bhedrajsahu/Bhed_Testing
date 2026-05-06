import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "meta_learner_nn_best.keras",
        compile=False,
        safe_mode=False
    )

    return model

model = load_model()

# -----------------------------------
# Class Labels
# -----------------------------------
class_names = [
    "Abnormal Heartbeat",
    "Covid-19",
    "MI",
    "MI History",
    "Normal"
]

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("ECG Image Classification")

st.write("Upload an ECG image for prediction")

uploaded_file = st.file_uploader(
    "Choose an ECG image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction
# -----------------------------------
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(image, caption="Uploaded ECG Image", use_container_width=True)

    # Resize image
    img = image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = float(prediction[0][predicted_class])

    # Show result
    st.success(f"Prediction: {class_names[predicted_class]}")

    st.info(f"Confidence: {confidence:.4f}")

    # Show probabilities
    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob:.4f}")
