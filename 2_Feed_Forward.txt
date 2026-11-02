
# Practical: Feed Forward Neural Network using TensorFlow and Keras

# Step 1: Import necessary packages

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)

# Step 2: Load MNIST training and testing data

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("\nTraining images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create training and validation datasets
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print("\nTraining samples:", len(x_train))
print("Validation samples:", len(x_val))
print("Testing samples:", len(x_test))

# Step 3: Define the Feed Forward Neural Network

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    # Dropout reduces overfitting
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])

# Display network architecture
print("\nModel Summary:")
model.summary()

# Step 4: Compile the model using SGD optimizer

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the neural network
print("\nTraining the model...")

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

# Step 5: Evaluate the network

print("\nEvaluating the model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# Evaluate training accuracy

train_loss, train_accuracy = model.evaluate(
    x_train,
    y_train,
    verbose=0
)

print("Training Accuracy:", round(train_accuracy * 100, 2), "%")

# Step 6: Plot training and validation accuracy

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()

# Plot training and validation loss

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# Step 7: Predict handwritten digits

predictions = model.predict(x_test[:10], verbose=0)

predicted_labels = np.argmax(predictions, axis=1)

print("\nDigit Predictions:")

for i in range(10):
    print(
        "Actual Digit:", y_test[i],
        "| Predicted Digit:", predicted_labels[i],
        "| Confidence:",
        round(float(np.max(predictions[i])) * 100, 2), "%"
    )

# Display sample handwritten digit predictions

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i], cmap="gray")

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("digit_predictions.png")
plt.show()

# Step 8: Generate confusion matrix

all_predictions = model.predict(x_test, verbose=0)

predicted_classes = np.argmax(all_predictions, axis=1)

cm = confusion_matrix(y_test, predicted_classes)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

display.plot(cmap="Blues", values_format="d")

plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Step 9: Display final result

if test_accuracy >= 0.90:
    print("\nSUCCESS: Test accuracy is greater than or equal to 90%.")
else:
    print("\nTest accuracy is below 90%.")
    print("Try training for more epochs.")

print("\nPractical completed successfully!")
