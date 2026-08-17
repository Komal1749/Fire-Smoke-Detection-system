import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# DATASET PATH
# =========================

train_path = "dataset/train"
test_path = "dataset/test"

# =========================
# IMAGE SETTINGS
# =========================

IMG_SIZE = (128, 128)
BATCH_SIZE = 64

# =========================
# DATA AUGMENTATION
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# TRAIN DATA
# =========================

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# =========================
# VALIDATION DATA
# =========================

validation_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =========================
# TEST DATA
# =========================

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Classes:", train_data.class_indices)

# =========================
# CNN MODEL
# =========================

model = Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(3, activation="softmax")
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/fire_smoke_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5,
    callbacks=[early_stop, checkpoint]
)

# =========================
# TEST MODEL
# =========================

loss, accuracy = model.evaluate(test_data)

print("\n==============================")
print("Final Test Accuracy:", accuracy)
print("==============================")

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# =========================
# LOSS GRAPH
# =========================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()

print("\nTraining Completed Successfully!")
print("Best model saved at: model/fire_smoke_model.keras")