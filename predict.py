import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# FIRE & SMOKE DETECTION - MOBILENETV2
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fire_smoke_mobilenetv2.keras"
)

IMAGE_SIZE = (128, 128)

# Model is loaded only when prediction is actually requested
model = None


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():

    global model

    if model is None:

        print("======================================")
        print("Loading MobileNetV2 model...")
        print("======================================")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("======================================")
        print("MobileNetV2 Model Loaded Successfully")
        print("======================================")

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    loaded_model = get_model()

    # Load image
    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    img_array = preprocess_input(
        img_array
    )

    # Prediction
    prediction = loaded_model.predict(
        img_array,
        verbose=0
    )

    # ========================================================
    # CLASS HANDLING
    # ========================================================

    # If model has 3 output classes
    if prediction.shape[-1] == 3:

        class_index = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            prediction[0][class_index]
        )

        classes = [
            "Smoke",
            "fire",
            "nonfire"
        ]

        predicted_class = classes[class_index]

    # If model has 2 output classes
    else:

        probability = float(
            prediction[0][0]
        )

        if probability >= 0.5:

            predicted_class = "fire"
            confidence = probability

        else:

            predicted_class = "nonfire"
            confidence = 1 - probability

    return predicted_class, confidence