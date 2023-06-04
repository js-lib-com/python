from keras.layers import Embedding

vocab_size = 10000  # size of the vocabulary
embedding_size = 32  # dimensionality of the embedding vectors

embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size)

input_sequence = [1, 2, 3, 4, 5]  # a sequence of integers

embedded_sequence = embedding_layer(input_sequence)

print(embedded_sequence.shape)  # prints (5, 32)
