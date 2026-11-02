
# Practical 5: Implement Continuous Bag of Words (CBOW) Model

# ==================================================
# A. DATA PREPARATION
# ==================================================

import numpy as np
import tensorflow as tf
import re
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Sample text corpus (8 sentences)

corpus = [
    "Natural language processing helps computers understand human language.",
    "Machine learning is an important part of artificial intelligence.",
    "Deep learning uses neural networks to learn complex patterns.",
    "Natural language processing uses machine learning techniques.",
    "Word embeddings represent words as numerical vectors.",
    "The CBOW model predicts a target word using surrounding context words.",
    "Machine learning and deep learning are useful in language processing.",
    "Word embeddings help computers understand relationships between words."
]


# Preprocess text

corpus = [re.sub(r"[^a-z\s]", "", sentence.lower()) for sentence in corpus]

print("Original Dataset:")

for sentence in corpus:
    print(sentence)


print("\nTotal number of sentences:", len(corpus))


# Tokenize the text

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index

vocab_size = len(word_index) + 1

print("\nWord Index:")
print(word_index)

print("\nTotal number of unique words:", len(word_index))
print("Total vocabulary size including padding index:", vocab_size)


# Convert sentences into sequences of integers

sequences = tokenizer.texts_to_sequences(corpus)

print("\nTokenized Sequences:")
for sequence in sequences:
    print(sequence)


# ==================================================
# B. GENERATE CBOW TRAINING DATA
# ==================================================

window_size = 2

contexts = []
targets = []

total_length = window_size * 2


# Generate context and target word pairs

for sequence in sequences:

    for i, word in enumerate(sequence):

        begin = max(0, i - window_size)
        end = min(len(sequence), i + window_size + 1)

        context = sequence[begin:i] + sequence[i + 1:end]

        contexts.append(context)
        targets.append(word)


# Pad context sequences to fixed length

contexts = pad_sequences(
    contexts,
    maxlen=total_length,
    padding="pre",
    truncating="pre"
)

contexts = np.array(contexts, dtype="int32")
targets = np.array(targets, dtype="int32")


# Convert target words into one-hot vectors

targets_one_hot = tf.keras.utils.to_categorical(
    targets,
    num_classes=vocab_size
)


print("\nContext-Target Training Data Generated!")

print("Input shape (X):", contexts.shape)
print("Output shape (y):", targets_one_hot.shape)


# Display sample context and target pairs

index_word = {
    index: word for word, index in word_index.items()
}

print("\nSample Context and Target Pairs:")

for i in range(5):

    context_words = [
        index_word[j] for j in contexts[i] if j != 0
    ]

    target_word = index_word[targets[i]]

    print("Context words:", context_words)
    print("Target word:", target_word)
    print("----------------------------------------")


# ==================================================
# C. BUILD AND TRAIN THE CBOW MODEL
# ==================================================

embedding_dim = 100


# Define input layer

context_input = Input(
    shape=(total_length,),
    dtype="int32"
)


# Embedding layer

embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    mask_zero=False
)

embedded_words = embedding_layer(context_input)


# Lambda layer to calculate average context embeddings
# Padding values (0) are excluded from the average

def average_embeddings(inputs):

    embeddings, token_ids = inputs

    mask = tf.cast(
        tf.not_equal(token_ids, 0),
        embeddings.dtype
    )

    mask = tf.expand_dims(mask, axis=-1)

    summed_embeddings = tf.reduce_sum(
        embeddings * mask,
        axis=1
    )

    valid_word_count = tf.reduce_sum(
        mask,
        axis=1
    )

    return tf.math.divide_no_nan(
        summed_embeddings,
        valid_word_count
    )


average_context = Lambda(
    average_embeddings,
    output_shape=(embedding_dim,)
)([embedded_words, context_input])


# Dense output layer with Softmax

output = Dense(
    vocab_size,
    activation="softmax"
)(average_context)


# Create CBOW model

model = Model(
    inputs=context_input,
    outputs=output,
    name="CBOW_Model"
)


# Compile the model

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nCBOW Model Summary:")
model.summary()


# Train the CBOW model

print("\nTraining the CBOW Model...")

history = model.fit(
    contexts,
    targets_one_hot,
    epochs=100,
    batch_size=16,
    verbose=1
)


print("\nModel Training Completed!")


# ==================================================
# D. MODEL EVALUATION
# ==================================================

loss, accuracy = model.evaluate(
    contexts,
    targets_one_hot,
    verbose=0
)


print("\nModel Evaluation:")
print("Loss:", round(loss, 4))
print("Accuracy:", round(accuracy * 100, 2), "%")


# ==================================================
# E. OUTPUT: PREDICT THE TARGET WORD
# ==================================================

sample_context = contexts[0:1]

predictions = model.predict(
    sample_context,
    verbose=0
)

predicted_index = np.argmax(predictions[0])

actual_index = targets[0]


print("\n--- CBOW Output ---")

print(
    "Context Words:",
    [index_word[i] for i in sample_context[0] if i != 0]
)

print("Actual Target Word:", index_word[actual_index])

print("Predicted Target Word:", index_word[predicted_index])


# ==================================================
# F. DISPLAY AND SAVE WORD EMBEDDINGS
# ==================================================

embedding_weights = embedding_layer.get_weights()[0]


print("\nWord Embedding for 'learning':")

print(embedding_weights[word_index["learning"]])


# Save word vectors in Word2Vec text format

vector_file = "vectors.txt"

with open(vector_file, "w", encoding="utf-8") as file:

    file.write(
        "{} {}\n".format(vocab_size - 1, embedding_dim)
    )

    for word, index in word_index.items():

        vector = " ".join(
            map(str, embedding_weights[index])
        )

        file.write("{} {}\n".format(word, vector))


print("\nWord vectors saved successfully!")

print("File location:", os.path.abspath(vector_file))


# ==================================================
# G. LOAD WORD VECTORS USING GENSIM
# ==================================================

try:

    from gensim.models import KeyedVectors

    cbow_output = KeyedVectors.load_word2vec_format(
        vector_file,
        binary=False
    )

    print("\nWord vectors loaded successfully using Gensim!")

    print("\nWords similar to 'learning':")

    similar_words = cbow_output.most_similar(
        positive=["learning"],
        topn=5
    )

    for word, similarity in similar_words:

        print(word, ":", round(similarity, 4))


except Exception as e:

    print("\nGensim could not load the vectors.")
    print("Install or check Gensim:", e)


# ==================================================
# FINAL RESULT
# ==================================================

print("\n========================================")
print("CBOW EXPERIMENT COMPLETED SUCCESSFULLY")
print("========================================")

print("Number of sentences:", len(corpus))
print("Unique words:", len(word_index))
print("Vocabulary size:", vocab_size)
print("Embedding dimensions:", embedding_dim)
print("Window size:", window_size)
print("Training examples:", len(contexts))
print("Final training accuracy:", round(accuracy * 100, 2), "%")
print("Word vector file:", vector_file)
