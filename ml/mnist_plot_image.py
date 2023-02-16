import numpy as np
import pandas
import matplotlib.pylab as plot

train_set = pandas.read_csv("train.csv")

print(train_set.head(0))

labels = train_set["label"]
print(labels)

train_set = train_set.drop("label", axis=1)
print(train_set)

print(train_set.shape)
print(labels.shape)

series = train_set.iloc[3]
grid_data = series.values.reshape(28, 28)

plot.figure(figsize=(1, 1))
plot.imshow(grid_data, interpolation="none", cmap="gray")
plot.show()
