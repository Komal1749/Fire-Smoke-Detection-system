import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ==========================================
# PROJECT SETTINGS
# ==========================================

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fire_smoke_mobilenetv2.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# DATA GENERATORS
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ==========================================
# TRAIN DATA
# ==========================================

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


# ==========================================
# VALIDATION DATA
# ==========================================

validation_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# ==========================================
# TEST DATA
# ==========================================

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print("\n==============================")
print("Classes:", train_data.class_indices)
print("==============================")

expected_classes = {"fire", "nonfire", "Smoke"}

actual_classes = set(train_data.class_indices.keys())

if actual_classes != expected_classes:
    raise ValueError(
        f"Dataset class mismatch!\n"
        f"Expected: {expected_classes}\n"
        f"Found: {actual_classes}"
    )

print("✅ All 3 classes found successfully!")

NUM_CLASSES = len(train_data.class_indices)

print("Number of Classes:", NUM_CLASSES)


# ==========================================
# PRETRAINED MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)


# Freeze pretrained layers
base_model.trainable = False


# ==========================================
# CLASSIFICATION HEAD
# ==========================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ==========================================
# TRAIN
# ==========================================

print("\n==============================")
print("Starting MobileNetV2 Training")
print("==============================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# ==========================================
# TEST
# ==========================================

loss, accuracy = model.evaluate(
    test_data
)

print("\n==============================")
print("MobileNetV2 Test Accuracy:",
      accuracy)
print("==============================")


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "MobileNetV2 Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    "mobilenet_accuracy_graph.png"
)

plt.show()


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "MobileNetV2 Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    "mobilenet_loss_graph.png"
)

plt.show()


# ==========================================
# COMPLETED
# ==========================================

print("\n================================")
print("Pretrained Model Training Done!")
print("Model saved at:")
print(MODEL_PATH)
print("================================")
# =========================================================
# CONFUSION MATRIX + CLASSIFICATION REPORT
# =========================================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


print("\n==============================")
print("Generating Classification Report")
print("==============================")


# Get predictions
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

class_names = list(
    test_data.class_indices.keys()
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names
)

print("\nClassification Report:")
print(report)


# Save report
with open(
    "classification_report.txt",
    "w"
) as f:

    f.write(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)


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

plt.show()


print("\n==============================")
print("Confusion Matrix Saved!")
print("==============================")

print("File:")
print("confusion_matrix.png")

print("\nClassification report saved:")
print("classification_report.txt")