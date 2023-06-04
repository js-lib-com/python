from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

word_vectors = KeyedVectors.load_word2vec_format("file://d:/tmp/model_lwcase_no_diac.bin", binary=True)
print(word_vectors)

calculator = word_vectors["calculator"]
computer = word_vectors["computer"]
keyboard = word_vectors["tastatura"]
display = word_vectors["monitor"]
mouse = word_vectors["mouse"]
pad = word_vectors["pad"]

human = word_vectors["om"]
man = word_vectors["barbat"]
woman = word_vectors["femeie"]
animal = word_vectors["animal"]
dog = word_vectors["caine"]
cat = word_vectors["pisica"]

road = word_vectors["sosea"]
highway = word_vectors["autostrada"]

q = dog - man + woman

distance = np.dot(computer, keyboard) / (np.linalg.norm(computer) * np.linalg.norm(keyboard))
print(f"Cosine distance between 'computer' and 'keyboard': {distance}")

distance = np.dot(dog, cat) / (np.linalg.norm(dog) * np.linalg.norm(cat))
print(f"Cosine distance between 'dog' and 'cat': {distance}")

distance = np.dot(q, cat) / (np.linalg.norm(q) * np.linalg.norm(cat))
print(f"Cosine distance between 'q' and 'cat': {distance}")

distance = np.dot(computer, cat) / (np.linalg.norm(computer) * np.linalg.norm(cat))
print(f"Cosine distance between 'computer' and 'cat': {distance}")

distance = np.dot(computer, road) / (np.linalg.norm(computer) * np.linalg.norm(road))
print(f"Cosine distance between 'computer' and 'road': {distance}")

distance = np.dot(computer, calculator) / (np.linalg.norm(computer) * np.linalg.norm(calculator))
print(f"Cosine distance between 'computer' and 'calculator': {distance}")

pca = PCA(n_components=3)
words = pca.fit_transform(
    [calculator, computer, keyboard, display, mouse, dog, cat, road, highway, pad, human, man, woman, animal, q])

X = [v[0] for v in words]
Y = [v[1] for v in words]
Z = [v[2] for v in words]

figure = plot.figure()
canvas = figure.canvas

subplot = figure.add_subplot(projection='3d')
scatter = subplot.scatter(X, Y, Z)

subplot.text(X[0], Y[0], Z[0], "calculator")
subplot.text(X[1], Y[1], Z[1], "computer")
subplot.text(X[2], Y[2], Z[2], "keyboard")
subplot.text(X[3], Y[3], Z[3], "display")
subplot.text(X[4], Y[4], Z[4], "mouse")
subplot.text(X[5], Y[5], Z[5], "dog")
subplot.text(X[6], Y[6], Z[6], "cat")
subplot.text(X[7], Y[7], Z[7], "road")
subplot.text(X[8], Y[8], Z[8], "highway")
subplot.text(X[9], Y[9], Z[9], "pad")
subplot.text(X[10], Y[10], Z[10], "human")
subplot.text(X[11], Y[11], Z[11], "man")
subplot.text(X[12], Y[12], Z[12], "woman")
subplot.text(X[13], Y[13], Z[13], "animal")
subplot.text(X[14], Y[14], Z[14], "q")

plot.show()
