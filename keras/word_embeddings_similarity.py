import numpy as np
from gensim.models import KeyedVectors
from gensim.models import FastText

# word_vectors = KeyedVectors.load_word2vec_format("file://d:/tmp/model_lwcase_no_diac.bin", binary=True)
# word_vectors = KeyedVectors.load_word2vec_format("file://d:/tmp/cc.en.300.vec.bin", binary=True)
word_vectors = KeyedVectors.load_word2vec_format("file://d:/tmp/cc.ro.300.vec.bin", binary=True)
# word_vectors = FastText.load_fasttext_format("file://d:/tmp/cc.en.300.bin")
print(type(word_vectors))
print(word_vectors)


# def get_top_similar(word: str, top: int = 10):
#     word_vector = word_vectors[word]
#
#     print(word_vector.shape)
#     word_vector = np.reshape(word_vector, (300, 1))
#     print(word_vector.shape)
#
#     dists = (word_vectors.vectors @ word_vector).flatten()
#     print(dists.shape)
#     print(dists)
#
#     top_word_indices = np.argsort(-dists)[1:top+1]
#     print(top_word_indices)
#     for index in top_word_indices:
#         similar_word = word_vectors.index_to_key[index]
#         print(similar_word)
#
#
# get_top_similar('green', 100)

print()
similar_words = word_vectors.most_similar("bǎrbat", topn=100)
for word, score in similar_words:
    print(word, score)

print()
king = word_vectors["rege"]
man = word_vectors["barbat"]
woman = word_vectors["femeie"]
vector = king - man + woman

similar_words = word_vectors.most_similar(positive=[vector], topn=100)
for word, score in similar_words:
    print(word, score)
