import gensim
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def add_position_encoding(word_embeddings):
    sentence_length, embedding_size = word_embeddings.shape
    position_encoding = np.zeros((sentence_length, embedding_size))
    for pos in range(sentence_length):
        for i in range(embedding_size):
            if i % 2 == 0:
                position_encoding[pos, i] = np.sin(pos / (10000 ** (i / embedding_size)))
            else:
                position_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / embedding_size)))
    return word_embeddings + position_encoding


def calculate_attention_weights(sentence_vector, word_embeddings):
    dot_product = np.dot(sentence_vector, word_embeddings.T)
    attention_weights = np.exp(dot_product - np.max(dot_product))
    attention_weights /= np.sum(attention_weights)
    return attention_weights


def calculate_sentence_vector(sentence, word2vec_model):
    sentence_words = sentence.split()
    sentence_length = len(sentence_words)
    word_embeddings = np.zeros((sentence_length, 300))
    for i, word in enumerate(sentence_words):
        if word in word2vec_model.vocab:
            word_embeddings[i, :] = word2vec_model[word]
    word_embeddings = add_position_encoding(word_embeddings)
    attention_weights = calculate_attention_weights(word_embeddings.mean(axis=1), word_embeddings)
    sentence_vector = np.dot(attention_weights, word_embeddings)
    return sentence_vector


def build_model(input_shape, output_shape):
    model = None
    pass


word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('file://d:/tmp/model_lwcase_no_diac.bin', binary=True)
