import numpy as np

# Define the word embeddings matrix with n words and d dimensions
n = 5  # number of words
d = 300  # dimensions of word embeddings
word_embeddings = np.random.rand(n, d)

# Initialize random query, key, and value vectors for each word
queries = np.random.rand(n, d)
keys = np.random.rand(n, d)
values = np.random.rand(n, d)

# Compute dot products between queries and keys for all pairs of words
dot_products = np.dot(queries, keys.T)

# Apply softmax to the dot products to obtain self-attention weights
attention_weights = np.exp(dot_products) / np.sum(np.exp(dot_products), axis=1, keepdims=True)

# Compute the self-attention weighted values by multiplying values with attention weights and summing up
weighted_values = np.dot(attention_weights, values)

# Output the final self-attention vectors
self_attention_vectors = weighted_values
print(self_attention_vectors)



import numpy as np

# Compute the self-attention matrix A
X = ... # A matrix of shape (n, d)
W = np.dot(X, X.T) / np.sqrt(d)
A = np.dot(W, W.T)
A = np.exp(A - np.max(A, axis=-1, keepdims=True))
A = A / np.sum(A, axis=-1, keepdims=True)
# return A

