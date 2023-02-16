import pickle
import matplotlib.pylab as plot

with open("mnist.pkl", "rb") as file:
    train_data = pickle.load(file, encoding="latin1")

train_set = train_data[0]
train_images = train_set[0]
train_labels = train_set[1]

figure = plot.figure(figsize=(6, 5))

for index in range(24):
    # train images are arrays of floating values normalized in range [0.0 .. 1.0]
    # plot.imshow knows to convert normalized value into gray scale byte
    # but first need to convert training image array into grid of 28 x 28

    train_image = train_images[index]
    pixels = train_image.reshape(28, 28)

    figure.add_subplot(4, 6, index + 1)
    plot.axis("off")
    plot.imshow(pixels, cmap="Greys")
    plot.title(train_labels[index])

plot.show()
