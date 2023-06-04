from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM, Dense
from keras.models import Model
import keras.preprocessing.text

# Define your text data as a list of strings
text_data = ["This is an example sentence.", "Here's another example sentence.",
             "The first sentence; the forth after semicolon.",
             "It seems semicolon is not considered as separator for sentence. Neither full stop character."]

sentences = [keras.preprocessing.text.text_to_word_sequence(sentence) for sentence in text_data]
print(sentences)

max_sentence_length = max([len(sentence) for sentence in sentences])
print(max_sentence_length)
padded_sequences = pad_sequences(sentences, maxlen=max_sentence_length, dtype=object, padding='post')
print(padded_sequences)

# Define your model architecture with an Embedding layer
input_layer = Input(shape=(max_sentence_length,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sentence_length)(
    input_layer)
lstm_layer = LSTM(64)(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(lstm_layer)
model = Model(inputs=input_layer, outputs=output_layer)
print(model.layers)

# Train the model with the padded sequences
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, [0, 1], epochs=1, batch_size=2)
