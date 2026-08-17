import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# FIRE & SMOKE DETECTION - MOBILENETV2
# ============================================================


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "model/fire_smoke_mobilenetv2.keras"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model = tf.keras.models.load_model(MODEL_PATH)

    print("======================================")
    print("MobileNetV2 Model Loaded Successfully")
    print("======================================")

except Exception as e:

    print("❌ Error loading model:")
    print(e)

    model = None


# ============================================================
# CLASS NAMES
# IMPORTANT:
# These must match train_data.class_indices
# ============================================================

classes = [
    "Smoke",
    "fire",
    "nonfire"
]


# ============================================================
# IMAGE SIZE
# ============================================================

IMG_SIZE = (128, 128)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = "MobileNetV2"
TEST_ACCURACY = 83.91


# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_image(img_path):

    # Check model
    if model is None:
        raise RuntimeError("Model could not be loaded.")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    # --------------------------------------------------------
    # Convert image to NumPy array
    # --------------------------------------------------------

    img = image.img_to_array(img)

    # --------------------------------------------------------
    # Add batch dimension
    # Shape:
    # (128,128,3)
    #        ↓
    # (1,128,128,3)
    # --------------------------------------------------------

    img = np.expand_dims(img, axis=0)

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------------------------

    img = preprocess_input(img)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        img,
        verbose=0
    )

    # --------------------------------------------------------
    # Get prediction probabilities
    # --------------------------------------------------------

    probabilities = prediction[0]

    # --------------------------------------------------------
    # Get predicted class index
    # --------------------------------------------------------

    class_index = np.argmax(probabilities)

    # --------------------------------------------------------
    # Get class name
    # --------------------------------------------------------

    predicted_class = classes[class_index]

    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities[class_index]
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return predicted_class, confidence


# ============================================================
# OPTIONAL TEST
# Run this file directly to test one image
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("Fire & Smoke Detection AI")
    print("Model:", MODEL_NAME)
    print("Test Accuracy:", TEST_ACCURACY, "%")
    print("Classes:", classes)
    print("======================================")