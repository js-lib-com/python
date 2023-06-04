import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load preprocessed ATIS dataset
train_df = pd.read_csv('atis_intents_train.csv')
print(train_df.columns)
print(train_df['intent'])
print(train_df['query'])

test_df = pd.read_csv('atis_intents_test.csv')
print(test_df.columns)
print(test_df['intent'])
print(test_df['query'])

# Preprocess input data
max_words = 1000
max_len = 150
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['query'])

X_train = tokenizer.texts_to_sequences(train_df['query'])
X_test = tokenizer.texts_to_sequences(test_df['query'])
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# Preprocess output labels
y_train = pd.get_dummies(train_df['intent']).values
y_test = pd.get_dummies(test_df['intent']).values
labels = y_train.shape[1]

# Define model architecture
model = keras.Sequential()
model.add(layers.Embedding(max_words, 20, input_length=max_len))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(labels, activation='softmax'))

model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
batch_size = 32
epochs = 20
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

# Evaluate model on test data
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# input_data = np.array(["a flight from Jassy to Bucharest, please"])
# prediction = model.predict(input_data, verbose=1)
# print(prediction)

# Preprocess new input statement
new_statement = 'i would like to fly from denver to pittsburgh on united airlines'
new_statement = 'i would like to find a flight from charlotte to las vegas that makes a stop in st. louis'
new_statement = 'on april first i need a ticket from tacoma to san jose departing before 7 am'
new_statement_seq = tokenizer.texts_to_sequences([new_statement])
new_statement_padded = keras.preprocessing.sequence.pad_sequences(new_statement_seq, maxlen=max_len)

# Make prediction
prediction = model.predict(new_statement_padded)
intent = np.argmax(prediction)

# Print predicted intent
print(prediction)
print(intent)
print(train_df['intent'].unique())
print('Predicted intent:', train_df['intent'].unique()[intent])
