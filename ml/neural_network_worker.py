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


class NeuralNetworkWorker(object):
    """
    A specialized feedforward network designed to make prediction based on input data. It is configurable via meta
    file - network descriptor, and loads its parameters from external binary file.
    """

    def __init__(self, config, data_file):
        assert config.type == "feedforward"

        self.layers = []
        self.input_size = config.input_vector

        input_size = self.input_size
        for layer_config in config.layers:
            self.layers.append(Layer(layer_config, input_size))
            input_size = layer_config.size

        with open(data_file, "rb") as file:
            for layer in self.layers:
                layer.load(file)

    def forward_propagation(self, input_data):
        assert input_data.shape == (self.input_size, 1)

        vector = input_data
        for layer in self.layers:
            vector = layer.activation(np.dot(layer.weights, vector) + layer.biases)
        return vector
