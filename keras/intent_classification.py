import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

filters_count = 64
kernel_size = 3
pool_size = 3

# Define training data
train_texts = [
    'book a table for two',
    'show me the menu',
    'cancel my reservation',
    'what is the phone number',
    'how much does it cost',
    'when are you open',
    'is there a dress code',
    'thank you for your help',
]
train_labels = [
    'reservation',
    'menu',
    'cancel_reservation',
    'phone_number',
    'cost',
    'hours',
    'dress_code',
    'thanks',
]

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1

# Convert training data to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
max_length = max([len(seq) for seq in train_sequences])
train_data = pad_sequences(train_sequences, maxlen=max_length, padding='post')

# Convert labels to one-hot encoded vectors
label_dict = {label: index for index, label in enumerate(set(train_labels))}
train_labels = [label_dict[label] for label in train_labels]
train_labels = to_categorical(train_labels)

# Define model architecture
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_length))
model.add(Conv1D(filters_count, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size))
model.add(Conv1D(filters_count, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size))
model.add(Flatten())
model.add(Dense(filters_count, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_dict), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_data, train_labels, epochs=50, batch_size=8)

# Test model
test_texts = [
    'do you have outdoor seating',
    'can I order takeout',
    'what time do you close',
]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=max_length, padding='post')
preds = model.predict(test_data)
pred_labels = [list(label_dict.keys())[list(label_dict.values()).index(np.argmax(pred))] for pred in preds]
print('Predictions:', pred_labels)
