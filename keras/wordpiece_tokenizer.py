import tensorflow as tf
import tensorflow_text as text

# vocabulary is a file provided by BERT model with a word (word piece) per line
VOCABULARY_PATH = "D:/jupyter/tensor-flow/hub/bert_en_uncased_L-8_H-512_A-8/model.2/assets/vocab.txt"
with open(VOCABULARY_PATH, "r", encoding="utf-8") as file:
    vocabulary = [line.strip() for line in file]

print(vocabulary[:100])
print(len(vocabulary))
print(tf.range(10))

# initializer keys are words from vocabulary whereas values are token value (actually index into vocabulary array)
vocab_init = tf.lookup.KeyValueTensorInitializer(vocabulary, tf.range(len(vocabulary), dtype=tf.int64))
table = tf.lookup.StaticHashTable(vocab_init, default_value=-1)
tokenizer = text.WordpieceTokenizer(table)

tokens = tokenizer.tokenize("The cat is on the table.")
tf.print(tokens)
print(tokens)
