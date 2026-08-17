import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "model/fire_smoke_mobilenetv2.keras"
TEST_PATH = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# =========================================================
# LOAD MODEL
# =========================================================

print("\n==============================")
print("Loading MobileNetV2 Model")
print("==============================")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully!")


# =========================================================
# TEST DATA
# =========================================================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


class_names = list(test_data.class_indices.keys())

print("\n==============================")
print("Classes")
print("==============================")

print(test_data.class_indices)


# =========================================================
# MODEL EVALUATION
# =========================================================

print("\n==============================")
print("Evaluating Model")
print("==============================")

loss, accuracy = model.evaluate(
    test_data,
    verbose=1
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {loss:.4f}"
)


# =========================================================
# PREDICTIONS
# =========================================================

print("\n==============================")
print("Generating Predictions")
print("==============================")

test_data.reset()

predictions = model.predict(
    test_data,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_data.classes


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names
)

print(report)


# Save report

with open(
    "classification_report.txt",
    "w"
) as file:

    file.write(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# =========================================================
# CONFUSION MATRIX GRAPH
# =========================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Fire & Smoke Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=200
)

plt.close()


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n==============================")
print("EVALUATION COMPLETED")
print("==============================")

print("✅ confusion_matrix.png created")

print("✅ classification_report.txt created")

print(
    f"✅ Test Accuracy: {accuracy * 100:.2f}%"
)