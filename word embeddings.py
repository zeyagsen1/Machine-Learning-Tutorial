import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example sentences
sentences = [
    "Word embeddings are a powerful tool in natural language processing.",
    "Machine learning algorithms can benefit from word embeddings.",
    "Python is a popular programming language for data science.",
]

# Tokenize sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Convert sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to ensure they have the same length
padded_sequences = pad_sequences(sequences)

# Create a simple sequential model with an embedding layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=total_words, output_dim=10, input_length=padded_sequences.shape[1]),
    tf.keras.layers.Flatten(),  # Flatten the 2D tensor to 1D
    tf.keras.layers.Dense(1, activation='sigmoid')  # Example output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
