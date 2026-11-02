
# Practical: Image Classification using CNN and MNIST
# TensorFlow and Keras

# --------------------------------------------------
# Stage 1: Loading and Preprocessing Image Data
# --------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
tf.random.set_seed(42)

print("CNN Image Classification using MNIST")
print("TensorFlow Version:", tf.__version__)

# Load MNIST dataset

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("\nOriginal Training Images:", x_train.shape)
print("Original Testing Images:", x_test.shape)

# Normalize pixel values to range 0-1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN requires image channels: height, width, channels

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Processed Training Images:", x_train.shape)
print("Processed Testing Images:", x_test.shape)


# --------------------------------------------------
# Stage 2: Defining the CNN Model Architecture
# --------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Data augmentation for better generalization

    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05),

    # First convolution block

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert feature maps to a vector

    tf.keras.layers.Flatten(),

    # Fully connected layer

    tf.keras.layers.Dense(128, activation="relu"),

    # Dropout regularization

    tf.keras.layers.Dropout(0.3),

    # Output layer for digits 0-9

    tf.keras.layers.Dense(10, activation="softmax")
])

print("\nCNN Model Architecture:")
model.summary()

# Compile the model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# Stage 3: Training the CNN Model
# --------------------------------------------------

print("\nTraining CNN Model...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# Stage 4: Estimating Model Performance
# --------------------------------------------------

print("\nEvaluating CNN Model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n========== FINAL RESULTS ==========")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

if test_accuracy > 0.90:
    print("Target Accuracy > 90%: Achieved")
else:
    print("Target Accuracy > 90%: Not achieved")


# --------------------------------------------------
# Plot Training and Validation Accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cnn_accuracy.png")
plt.show()


# --------------------------------------------------
# Plot Training and Validation Loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cnn_loss.png")
plt.show()


# --------------------------------------------------
# Display Sample Predictions
# --------------------------------------------------

predictions = model.predict(x_test[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Digit Predictions:")

for i in range(10):
    confidence = np.max(predictions[i]) * 100

    print(
        f"Actual: {y_test[i]} | "
        f"Predicted: {predicted_labels[i]} | "
        f"Confidence: {confidence:.2f}%"
    )

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap="gray")

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("cnn_predictions.png")
plt.show()


# --------------------------------------------------
# Generate Confusion Matrix
# --------------------------------------------------

all_predictions = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(all_predictions, axis=1)

cm = confusion_matrix(y_test, predicted_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

disp.plot(cmap="Blues", values_format="d")

plt.title("CNN MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")
plt.show()

print("\nCNN Image Classification Practical Completed!")
