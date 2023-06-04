# from gensim.scripts.glove2word2vec import glove2word2vec
#
# input_file = "file://d:/tmp/cc.en.300.vec"
# output_file = "file://d:/tmp/cc.en.300.vec.bin"
# glove2word2vec(input_file, output_file)

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("d:/tmp/cc.ro.300.vec", binary=False)
model.save_word2vec_format("d:/tmp/cc.ro.300.vec.bin", binary=True)
