import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, GlobalMaxPooling1D, Conv1D
from keras.layers import Embedding, concatenate, SpatialDropout1D, Dot
from keras.utils import to_categorical


def positional_encoding(sentence_length, embedding_size):
    pos = np.arange(sentence_length)[:, np.newaxis]
    i = np.arange(embedding_size)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_size))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads[np.newaxis, ...]


def sentence_encoding(sentence, word2vec_model, max_sentence_length, embedding_size):
    sentence = sentence.lower().split()[:max_sentence_length]
    sentence_length = len(sentence)
    sentence_embedding = np.zeros((max_sentence_length, embedding_size))
    for i, word in enumerate(sentence):
        if word in word2vec_model:
            sentence_embedding[i] = word2vec_model[word]
    sentence_embedding *= np.sqrt(embedding_size)
    sentence_embedding += positional_encoding(sentence_length, embedding_size)
    return sentence_embedding


model_path = 'file://d:/tmp/model_lwcase_no_diac.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

dataset = [
    ("Hello", "Greeting"),
    ("Hi there", "Greeting"),
    ("How are you doing?", "HowAreYou"),
    ("What is the weather like today?", "Weather"),
    ("What is your name?", "Name"),
    ("What can you do?", "Capabilities"),
    ("Goodbye", "Goodbye"),
    ("See you later", "Goodbye"),
]

max_sentence_length = 10
embedding_size = 300
num_classes = 6
input_shape = (max_sentence_length, embedding_size)

encoded_dataset = []
for sentence, intent in dataset:
    sentence_encoding = sentence_encoding(sentence, word2vec_model, max_sentence_length, embedding_size)
    encoded_dataset.append((sentence_encoding, intent))

X = np.array([sentence_encoding for sentence_encoding, _ in encoded_dataset])
y = np.array([intent for _, intent in encoded_dataset])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 10

# convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)

model.fit(X_train, y_train_encoded,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test_encoded))
