import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

train_data = [
    ("book_flight", "I want to book a flight from Boston to New York"),
    ("book_flight", "I need to fly from San Francisco to Los Angeles"),
    ("cancel_reservation", "Can you help me cancel my hotel reservation?"),
    ("cancel_reservation", "I would like to cancel my reservation for next week"),
    ("find_promotions", "Are there any promotions available for the flights next month?"),
]

labels_map = {"book_flight": 0, "cancel_reservation": 1, "find_promotions": 2}

train_inputs = [
    tokenizer.encode_plus(data[1], add_special_tokens=True, max_length=128, padding='max_length', truncation=True,
                          return_tensors='tf') for data in train_data]
train_labels = [labels_map[data[0]] for data in train_data]

# train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).batch(2)

num_labels = len(labels_map)

input_ids_layer = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
input_masks_layer = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_masks')
input_segments_layer = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_segments')

train_dataset = tf.data.Dataset.from_tensor_slices(
    (input_ids_layer, input_masks_layer, input_segments_layer, train_labels))
# train_dataset = train_dataset.shuffle(100).batch(32).repeat(3)

bert_output = bert_model(
    input_ids_layer, attention_mask=input_masks_layer, token_type_ids=input_segments_layer
)[1]

dense_layer = tf.keras.layers.Dense(num_labels, activation='softmax')(bert_output)

model = tf.keras.models.Model(inputs=[input_ids_layer, input_masks_layer, input_segments_layer], outputs=dense_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_dataset, epochs=3)
