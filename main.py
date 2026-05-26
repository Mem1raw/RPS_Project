import numpy as np
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

SEED = 20
IMG_SIZE = (64, 64)

# Set environment variables for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Train Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\emirh\\PycharmProjects\\PythonProject\\RPS_Data\\Rock-Paper-Scissors\\train",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode='categorical',
)

# Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\emirh\\PycharmProjects\\PythonProject\\RPS_Data\\Rock-Paper-Scissors\\validation",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode='categorical',
)

# Test Dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\emirh\\PycharmProjects\\PythonProject\\RPS_Data\\Rock-Paper-Scissors\\test",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])


def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label


train_ds = train_ds.map(
    augment,
    num_parallel_calls=tf.data.AUTOTUNE
)

# --- MODEL ARCHITECTURE ---
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Block 4 (Extra depth for better feature extraction)
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Flatten instead of GlobalAveragePooling to retain spatial details for hand shapes
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Strong dropout here to prevent overfitting

    tf.keras.layers.Dense(3, activation='softmax'),
])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(
    optimizer='adam',  # Adam optimizer generally converges faster
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[callback]
)

model.summary()

test_loss, test_accuracy = model.evaluate(test_ds)

# --- REPORTING AND PERCENTAGE OUTPUTS ---
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\n--- Class-wise Accuracy ---")
for i, acc_score in enumerate(class_accuracy):
    print(f"{test_ds.class_names[i]}: %{acc_score * 100:.2f}")

print("\n--- Classification Report ---")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_ds.class_names
))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("\n--- Overall Results ---")
# Get the last element of the list (-1) and convert to percentage
print(f"Final Training Accuracy: %{acc[-1] * 100:.2f}")
print(f"Final Validation Accuracy: %{val_acc[-1] * 100:.2f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: %{test_accuracy * 100:.2f}")

# --- PLOTTING ---
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training, Validation, and Test Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training, Validation, and Test Loss')

plt.show()
plt.show()