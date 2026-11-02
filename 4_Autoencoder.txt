
# Practical: Anomaly Detection Using Autoencoder


# Step 1: Import required libraries

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix


# Step 2: Load the ECG dataset

url = "https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"


data = pd.read_csv(url, header=None)


print("Dataset Loaded Successfully")

print("Dataset Shape:", data.shape)

print(data.head())


# Step 3: Separate features and labels

X = data.iloc[:, :-1].values.astype("float32")

y = data.iloc[:, -1].values.astype("int32")


# Labels: 0 = Anomaly, 1 = Normal


# Step 4: Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train only on normal observations

X_train_normal = X_train[y_train == 1]


print("\nTraining data shape:", X_train_normal.shape)

print("Testing data shape:", X_test.shape)


# Step 5: Build the Encoder

input_dim = X.shape[1]


encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu")
], name="Encoder")


# Step 6: Build the Decoder

decoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="linear")
], name="Decoder")


# Step 7: Combine Encoder and Decoder

autoencoder = tf.keras.Sequential([
    encoder,
    decoder
], name="Autoencoder")


# Step 8: Compile the model

autoencoder.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nAutoencoder Model Summary:")

autoencoder.summary()


# Step 9: Train the Autoencoder

history = autoencoder.fit(
    X_train_normal,
    X_train_normal,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)


# Step 10: Calculate reconstruction errors

X_train_pred = autoencoder.predict(X_train_normal, verbose=0)


train_errors = np.mean(
    np.square(X_train_normal - X_train_pred), axis=1
)


# Set threshold using the 95th percentile of training errors

threshold = np.percentile(train_errors, 95)


print("\nAnomaly Threshold:", threshold)


# Step 11: Reconstruct test observations

X_test_pred = autoencoder.predict(X_test, verbose=0)


test_errors = np.mean(
    np.square(X_test - X_test_pred), axis=1
)


# Higher error indicates an anomaly

y_pred = np.where(test_errors > threshold, 0, 1)


# Step 12: Evaluate the model

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Anomaly", "Normal"],
    zero_division=0
))


# Step 13: Display sample predictions

results = pd.DataFrame({
    "Actual Label": y_test,
    "Reconstruction Error": test_errors,
    "Predicted Label": y_pred
})


print("\nSample Anomaly Detection Results:")

print(results.head(10).to_string(index=False))


# Step 14: Plot training and validation loss

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Mean Squared Error")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# Step 15: Plot reconstruction errors

plt.figure(figsize=(8, 5))

plt.hist(test_errors[y_test == 1], bins=40,
         alpha=0.7, label="Normal")

plt.hist(test_errors[y_test == 0], bins=40,
         alpha=0.7, label="Anomaly")

plt.axvline(threshold, linestyle="--",
            label="Anomaly Threshold")

plt.title("Reconstruction Errors for ECG Data")

plt.xlabel("Mean Squared Reconstruction Error")

plt.ylabel("Number of Observations")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


print("\nAnomaly Detection Completed Successfully!")
