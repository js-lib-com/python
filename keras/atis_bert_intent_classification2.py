import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Load the pre-trained BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the input and output dimensions
max_length = 128
num_labels = 5

# Define the input layer
input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_mask')
segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='segment_ids')

# Define the BERT layer
bert_layer = bert_model([input_ids, input_mask, segment_ids])[1]

# Define the output layer
output_layer = tf.keras.layers.Dense(num_labels, activation='softmax')(bert_layer)

# Define the model
model = tf.keras.models.Model(inputs=[input_ids, input_mask, segment_ids], outputs=output_layer)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the training data
train_data = [('book a flight from New York to London', 'book_flight'),
              ('reserve a table for two at 7pm', 'book_restaurant'),
              ('what is the weather like today?', 'weather'),
              ('play some music', 'play_music'),
              ('set a reminder for tomorrow at 10am', 'set_reminder')]

label_map = {'book_flight': 0, 'book_restaurant': 1, 'weather': 2, 'play_music': 3, 'set_reminder': 4}

train_inputs = []
train_labels = []
for data in train_data:
    train_inputs.append(
        tokenizer.encode_plus(data[0], add_special_tokens=True, max_length=max_length, pad_to_max_length=True,
                              return_attention_mask=True, return_token_type_ids=True))
    # train_labels.append(tf.one_hot([label_map[data[1]]], depth=num_labels)[0])
    train_labels.append(tf.one_hot([label_map[data[1]]], depth=num_labels))

training_inputs = [tf.convert_to_tensor([input_dict['input_ids']]) for input_dict in train_inputs]
train_masks = [tf.convert_to_tensor([input_dict['attention_mask']]) for input_dict in train_inputs]
train_segments = [tf.convert_to_tensor([input_dict['token_type_ids']]) for input_dict in train_inputs]

print(len(training_inputs))
print(len(train_masks))
print(len(train_segments))
print(train_labels)
print(len(train_labels))

# Train the model
model.fit(x=(training_inputs, train_masks, train_segments), y=train_labels, epochs=3)

# Make a prediction
test_input = tokenizer.encode_plus('reserve a seat for two at 7pm', add_special_tokens=True, max_length=max_length,
                                   pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True)
test_input = [tf.convert_to_tensor([test_input['input_ids']])]
test_mask = [tf.convert_to_tensor([test_input['attention_mask']])]
test_segment = [tf.convert_to_tensor([test_input['token_type_ids']])]
prediction = model.predict([test_input, test_mask, test_segment])
print(prediction)
