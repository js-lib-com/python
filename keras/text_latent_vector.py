from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

texts = ["I want to know how much I have to pay this month.", "I want to send my electricity meter.",
         "When I can send my gas meter this month?"]

tokenizer = Tokenizer(num_words=1000)  # set num_words to limit the number of unique words to consider
tokenizer.fit_on_texts(texts)  # texts is a list of strings representing your text data
sequences = tokenizer.texts_to_sequences(texts)  # convert text to a list of integer sequences
print(sequences)

embedding_layer = Embedding(input_dim=1000, output_dim=64, input_length=20)
print(embedding_layer)

config = embedding_layer.get_config()
print(config['input_dim'])  # Output: 1000
print(config['output_dim'])  # Output: 64
print(config['input_length'])  # Output: 10

embedded_sequences = embedding_layer(sequences)
