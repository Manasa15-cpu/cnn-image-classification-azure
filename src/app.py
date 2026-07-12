import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -----------------------------
# Load Model
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

IMG_SIZE = (150, 150)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):

    image = image.resize(IMG_SIZE)
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image.astype("float32") 
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions)
    confidence = float(predictions[0][predicted_index]) * 100

    return CLASS_NAMES[predicted_index], confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Intel Image Classifier", layout="centered")

st.title("🖼️ Intel Image Classification")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    predicted_class, confidence = predict_image(image)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")