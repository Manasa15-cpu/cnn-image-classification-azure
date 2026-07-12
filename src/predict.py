from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "intel_image_classifier.keras"

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

IMAGE_SIZE = (150, 150)


def predict_image(image_path: Path) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    predict_image(Path(sys.argv[1]))