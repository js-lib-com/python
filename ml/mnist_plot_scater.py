import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

with open("mnist.pkl", "rb") as file:
    data_set = pickle.load(file, encoding="latin1")

size = 4000

train_set = data_set[0]
train_images = train_set[0][0:size]
train_labels = train_set[1][0:size]
print(train_images.shape)

pca = PCA(n_components=3)
train_vectors = pca.fit_transform(train_images)
print(train_vectors.shape)

X = [v[0] for v in train_vectors]
Y = [v[1] for v in train_vectors]
Z = [v[2] for v in train_vectors]

colors = ["black", "red", "green", "blue", "gold", "brown", "purple", "magenta", "cyan", "slategray"]
C = [colors[label] for label in train_labels]


def on_button_press(event):
    contains, indices = scatter.contains(event)
    if contains:
        print([train_labels[index] for index in indices["ind"]])
    canvas.draw_idle()


figure = plot.figure()
canvas = figure.canvas
# canvas.mpl_connect("motion_notify_event", on_hover)
canvas.mpl_connect("button_press_event", on_button_press)

subplot = figure.add_subplot(projection='3d')
# subplot.axis("off")
scatter = subplot.scatter(X, Y, Z, c=C)
plot.show()
