import numpy as np
from gensim.models import KeyedVectors
from keras.activations import sigmoid
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from scipy.special import softmax
from unidecode import unidecode

aliases = {
    'anulez': 'anula',
    'returnez': 'returna',
    'invalidez': 'invalida',
}

word_embedding_size = 300
maximum_sentence_words = 30
input_vector_size = maximum_sentence_words * word_embedding_size

word_vectors = KeyedVectors.load_word2vec_format("file://d:/tmp/model_lwcase_no_diac.bin", binary=True)
print(word_vectors)


def tokenize(sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([sentence])
    return [unidecode(tokenizer.index_word[index]) for index in tokenizer.texts_to_sequences([sentence])[0]]


def self_attention_weights(X):
    """
    Compute self attention weights for a matrix of word embeddings. This solutions based on input data only, is does
    not uses weighting parameters that should be optimized by learning process. Note that self attention is a
    particular case of attention algorithm where Q = K = V and all equal with input sequence.

    :param X: (numpy.ndarray) an word embeddings matrix on [n x d] dimension, where 'n' is the number of words and 'd'
    is the word embeddings size.
    :return: self attention weights to apply to each word when compute sentence weighted words sum
    (sentence embeddings vector).
    """

    # commented out is alternative with self attention disabled
    # return np.full((X.shape[0], X.shape[0]), 1 / X.shape[0], dtype=np.float32)

    W = np.dot(X, X.T)
    return softmax(W, axis=1)


def sentence_vector(words):
    word_embeddings_matrix = np.zeros((maximum_sentence_words, word_embedding_size))
    words_count = 0
    for word in words:
        if words_count == maximum_sentence_words:
            break
        if word in aliases:
            word = aliases[word]
        if word in word_vectors:
            word_embeddings_matrix[words_count] = word_vectors[word]
            words_count += 1
        else:
            print(f'Word without embeddings: {word}.')

    if words_count == 0:
        return np.zeros(word_embedding_size)

    word_weights = self_attention_weights(word_embeddings_matrix)
    for i in range(words_count, maximum_sentence_words):
        word_weights[i] = np.zeros(word_weights.shape[1])
    return np.ravel(word_weights @ word_embeddings_matrix)


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

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.sentences):
            raise StopIteration
        item = (self.sentences[self.index], self.input_data[self.index], self.target_value[self.index])
        self.index += 1
        return item


train_set = DataSet('intent_classification_train_set')

model = Sequential()
model.add(Dense(40, input_dim=input_vector_size, activation=sigmoid))
model.add(Dense(train_set.classes_count, activation=sigmoid))

model.compile(optimizer=Adam(), loss=mean_squared_error, metrics=['accuracy'])
model.fit(train_set.input_data, train_set.target_value, epochs=60, batch_size=1, verbose=0)

test_set = DataSet('intent_classification_test_set')
for sentence, input_data, target_value in test_set:
    prediction = model.predict(np.array(input_data).reshape(1, input_vector_size), verbose=0)
    prediction_value = np.argmax(prediction)
    target_value = np.argmax(target_value)

    print()
    print(sentence)
    if prediction_value != target_value:
        print("ERROR!")
        print(prediction)
    print(f'target:{target_value}, prediction:{prediction_value}, confidence:{np.max(prediction)}')

# graph.scatter(range(y_train.shape[1]), prediction)
# graph.show()
