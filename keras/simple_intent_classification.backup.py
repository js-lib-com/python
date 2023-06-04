from gensim.models import KeyedVectors
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from unidecode import unidecode
import numpy as np
import matplotlib.pyplot as graph

aliases = {
    'anulez': 'anula',
    'returnez': 'returna',
    'invalidez': 'invalida',
}

word_embedding_size = 300
word_vectors = KeyedVectors.load_word2vec_format("file://d:/tmp/model_lwcase_no_diac.bin", binary=True)
print(word_vectors)


def tokenize(sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([sentence])
    return [unidecode(tokenizer.index_word[index]) for index in tokenizer.texts_to_sequences([sentence])[0]]


def positional_encoding(sentence_length):
    return np.zeros((sentence_length, word_embedding_size))
    pos = np.arange(sentence_length)[:, np.newaxis]
    i = np.arange(word_embedding_size)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(word_embedding_size))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads


def self_attention_weights(X):
    """
    Compute self attention weights for a matrix of word embeddings.
    
    :param X: (numpy.ndarray) an word embeddings matrix on [n x d] dimension, where 'n' is the number of words and 'd'
    is the word embeddings size.
    :return: self attention weights to apply to each word when compute sentence weighted words sum
    (sentence embeddings vector).
    """
    d = X.shape(1)
    W = np.dot(X, X.T) / np.sqrt(d)
    A = np.dot(W, W.T)
    A = np.exp(A - np.max(A, axis=-1, keepdims=True))
    A = A / np.sum(A, axis=-1, keepdims=True)
    return A


def sentence_vector(words):
    # sentence vector has the same size as word embedding vector
    vector = np.zeros(word_embedding_size)

    words_count = 0
    for word in words:
        if word in aliases:
            word = aliases[word]
        if word in word_vectors:
            words_count += 1

    if words_count == 0:
        return vector

    word_position_matrix = positional_encoding(words_count)
    i = 0
    for word in words:
        if word in aliases:
            word = aliases[word]

        if word in word_vectors:
            # applies word position on original word embedding, before adding weights
            word_vector = word_vectors[word] + word_position_matrix[i]
            vector = np.add(vector, word_vector, out=vector)
            i += 1
        else:
            print(f'Unknown embeddings for word {word}.')

    # divide by words count to create average weighting to avoid sentence scaling with its length
    return np.divide(vector, words_count, out=vector)


def to_categorical(classes):
    unique_classes = np.unique(np.array(classes))
    print(unique_classes)

    def hot_one(cls):
        hot_one = np.zeros(len(unique_classes))
        hot_one[np.where(unique_classes == cls)] = 1
        return hot_one

    return np.array([hot_one(cls) for cls in classes])


class DataSet(object):
    def __init__(self, file_name):
        self.sentences = []

        sentence_vectors = []
        classes = []
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                self.sentences.append(line.strip())
                tokens = tokenize(line)
                sentence_vectors.append(sentence_vector(tokens[1:]))
                classes.append(tokens[0])

        self.classes_count = len(np.unique(np.array(classes)))
        self.input_data = np.array(sentence_vectors)
        self.target_value = to_categorical(classes)


train_set = DataSet('intent_classification_train_set')

model = Sequential()
model.add(Dense(30, input_dim=word_embedding_size, activation="sigmoid"))
model.add(Dense(train_set.classes_count, activation="sigmoid"))

model.compile(optimizer=Adam(), loss=mean_squared_error, metrics=['accuracy'])
model.fit(train_set.input_data, train_set.target_value, epochs=60, batch_size=1, verbose=0)

test_set = DataSet('intent_classification_test_set')
for index in range(len(test_set.sentences)):
    prediction = model.predict(np.array(test_set.input_data[index]).reshape(1, word_embedding_size), verbose=0)
    target_value = np.argmax(test_set.target_value[index])
    prediction_value = np.argmax(prediction)

    print()
    print(test_set.sentences[index])
    if prediction_value != target_value:
        print("ERROR!")
        print(prediction)
    print(f'target:{target_value}, prediction:{prediction_value}, confidence:{np.max(prediction)}')

# graph.scatter(range(y_train.shape[1]), prediction)
# graph.show()
