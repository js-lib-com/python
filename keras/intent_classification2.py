import numpy as np
import gensim
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Input, Dot, Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# Load pre-trained Word2Vec model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/path/to/your/word2vec/model.bin', binary=True)

def create_sentence_vector(sentence, w2v_model, max_len=50):
    # Split sentence into words and remove stop words
    words = [w for w in sentence.split() if w not in stop_words]

    # Convert words to indices
    word_indices = [tokenizer.word_index[w] for w in words if w in tokenizer.word_index.keys()]

    # Pad or truncate to max_len
    padded_word_indices = pad_sequences([word_indices], maxlen=max_len, padding='post', truncating='post')[0]

    # Get the word embeddings for the word indices
    word_embeddings = np.array([w2v_model.wv[w] if w in w2v_model.wv.vocab else np.zeros(w2v_model.vector_size) for w in tokenizer.index_word.values()[:len(tokenizer.word_index)+1]])

    # Augment word embeddings with position encoding
    position_encoding = np.array([[pos / np.power(10000, 2 * i / w2v_model.vector_size) for i in range(w2v_model.vector_size)] for pos in range(max_len)])
    word_embeddings = np.add(word_embeddings, position_encoding)

    # Apply attention mechanism to the sentence vectors
    inputs = Input(shape=(max_len, w2v_model.vector_size))
    attention_weights = Dense(1, activation='tanh')(inputs)
    attention_weights = Flatten()(attention_weights)
    attention_weights = Activation('softmax')(attention_weights)
    attention_weights = RepeatVector(w2v_model.vector_size)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)
    sentence_vector = Dot(axes=[1, 1])([inputs, attention_weights])
    sentence_vector = Dense(w2v_model.vector_size, activation='relu')(sentence_vector)
    sentence_vector = Dropout(0.5)(sentence_vector)

    # Create and return the sentence vector
    model = Model(inputs=inputs, outputs=sentence_vector)
    sentence_vector = model.predict(np.array([word_embeddings[padded_word_indices]]))
    return sentence_vector[0]

# Define the model architecture
input_shape = (max_len, w2v_model.vector_size)
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, w2v_model.vector_size, input_length=max_len, weights=[word_embeddings]))
model.add(Dense(64, activation='relu'))
model.add(Flatten())
