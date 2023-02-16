import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pylab as plot

train_set = pandas.read_csv("train.csv")
labels = train_set["label"]
train_set = train_set.drop("label", axis=1)

scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)

pca = PCA(n_components=2)
train_set = pca.fit_transform(train_set)
train_set = train_set[0:10000]
print(train_set.shape)

colors = ["black", "red", "green", "blue", "gold", "brown", "purple", "magenta", "cyan", "yellow"]

for vector, label in zip(reversed(train_set), reversed(labels)):
    plot.scatter(vector[0], vector[1], color=colors[label])

plot.show()
