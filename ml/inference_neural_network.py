import pickle
import numpy as np
from munch import DefaultMunch
import util


class Layer(object):
    def __init__(self, config, input_size):
        self.weights = np.empty((config.size, input_size))
        self.biases = np.empty((config.size, 1))

        match config.activation.lower():
            case "relu":
                self.activation = util.relu

            case "sigmoid" | _:
                self.activation = util.sigmoid

    def load(self, file):
        for index in np.ndindex(self.weights.shape):
            self.weights[index] = util.bytes2float(file.read(4))

        for index in np.ndindex(self.biases.shape):
            self.biases[index] = util.bytes2float(file.read(4))


class InferenceNeuralNetwork(object):
    """
    A specialized feedforward network designed to make prediction based on input data. It is configurable via meta
    file - network descriptor, and loads its parameters from external binary file.
    """

    def __init__(self, config, data_file):
        print("Inference Neural Network")
        assert config.type == "feedforward"

        self.layers = []
        input_size = config.input
        for layer_config in config.layers:
            self.layers.append(Layer(layer_config, input_size))
            input_size = layer_config.size

        with open(data_file, "rb") as file:
            for layer in self.layers:
                layer.load(file)

    def forward_propagation(self, input_data):
        vector = input_data
        for layer in self.layers:
            vector = layer.activation(np.dot(layer.weights, vector) + layer.biases)
        return vector


def main():
    config = DefaultMunch.fromDict(util.load_config("demo.yml"))
    inn = InferenceNeuralNetwork(config, "demo.data")
    print(inn)

    with open("mnist.pkl", "rb") as file:
        data_set = pickle.load(file, encoding="latin1")

    test_set = data_set[2]
    test_images = test_set[0]
    test_labels = test_set[1]

    index = 1357
    input_data = np.array(test_images[index])
    print(input_data.shape)
    input_data = np.reshape(input_data, (784, 1))
    print(input_data.shape)

    result = inn.forward_propagation(input_data)
    print(np.argmax(result))
    print(test_labels[index])


main()
