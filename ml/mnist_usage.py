import pickle
import json
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Network(object):
    def __init__(self, json_file):
        data = json.load(json_file)
        self.biases = np.asarray(data["biases"])
        self.weights = np.asarray(data["weights"])

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


with open("network.json", "r", encoding="utf-8") as file:
    network = Network(file)

with open("mnist.pkl", "rb") as file:
    data_set = pickle.load(file, encoding="latin1")

test_set = data_set[2]
test_images = test_set[0]
test_labels = test_set[1]

index = 3
input_data = np.array(test_images[index])
print(input_data.shape)
input_data = np.reshape(input_data, (784, 1))
print(input_data.shape)

result = network.feed_forward(input_data)
print(result)
print(test_labels[index])
