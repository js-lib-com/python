import json
import numpy as np
from numpy import random
from util import cost_prime
from util import sigmoid
from util import sigmoid_prime
import time


class NeuralLayer(object):
    def __init__(self, layer_size, previous_layer_size):
        # weights matrix of the previous layer synapses; matrix dimensions layer_size x previous_layer_size
        self.weights = np.random.randn(layer_size, previous_layer_size)
        # neurons biases matrix of layer_size x 1 (column vector)
        self.biases = np.random.rand(layer_size, 1)
        assert self.weights.shape[0] == self.biases.shape[0]

        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

        # cache for last computed weighted inputs and activation values; updated by forward propagation
        self.weighted_inputs = np.zeros((layer_size, 1))
        self.activations = np.zeros((layer_size, 1))

        self.layer_type = "hidden"
        self.previous_layer = None
        self.next_layer = None

    def set_type(self, layer_type):
        self.layer_type = layer_type

    def forward_propagation(self, data_vector):
        # np.dot operates on matrices
        # data_vector parameter is a column vector (a matrix with a single column)
        # for matrix dot product to be possible the number of columns of the first matrix should be equal
        # with the number rows of the second matrix
        # resulting matrix has the number of rows from the first matrix and the number of columns from the second
        # in our case: [m x n] . [n x 1] -> [m x 1]

        assert self.weights.shape[1] == data_vector.shape[0]
        assert data_vector.shape[1] == 1

        np.dot(self.weights, data_vector, out=self.weighted_inputs)  # Z = W x X
        np.add(self.weighted_inputs, self.biases, out=self.weighted_inputs)  # Z = W x X + B
        sigmoid(self.weighted_inputs, out=self.activations)  # A = sigmoid(Z)
        return self.activations

    def train(self, cost_error, previous_layer_activations, learning_rate):
        assert previous_layer_activations.shape == (self.weights.shape[1], 1)

        # use back propagation algorithm to compute this layer cost error, using the next layer cost error
        # then use cost error to compute this layer weights and biases gradient

        np.dot(cost_error, previous_layer_activations.T, self.weights_gradient)
        np.multiply(self.weights_gradient, learning_rate, out=self.weights_gradient)
        np.subtract(self.weights, self.weights_gradient, out=self.weights)

        np.multiply(cost_error, learning_rate, self.biases_gradient)
        np.subtract(self.biases, self.biases_gradient, out=self.biases)

    # experimental
    def cost_error(self, prediction, target_value):
        if self.layer_type == "output":
            return cost_prime(prediction, target_value) * sigmoid_prime(self.weighted_inputs)

    def activation_prime(self):
        return sigmoid_prime(self.weighted_inputs)

    def dump(self, file):
        data = {"weights": self.weights, "biases": self.biases}
        json.dump(data, file)


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.layers = [NeuralLayer(layer_size, previous_layer_size) for layer_size, previous_layer_size in
                       zip(sizes[1:], sizes[:-1])]

        self.output_layer = self.layers[-1]
        self.output_layer.set_type("output")

        self.epoch_results = []

    def forward_propagation(self, data_vector):
        """
        Get data input vector and propagates it throw network layers. A layer output vector (aka activation vector) is
        used as input vector for the next layer. Returns the output vector of the last layer, that is usually
        named output layer.

        :param data_vector: input data as column vector (matrix with a single column),
        :return: network output vector is the activation vector or the last layer.
        """

        vector = data_vector
        for layer in self.layers:
            vector = layer.forward_propagation(vector)
        return vector

    def train(self, train_set, test_set, epochs, learning_rate):
        for epoch in range(epochs):
            start_timestamp = time.time()
            random.shuffle(train_set)
            activations = [np.array([])] + [layer.activations for layer in self.layers[:-1]]

            for input_data, target_value in train_set:
                activations[0] = input_data

                prediction = self.forward_propagation(input_data)
                assert prediction.shape == target_value.shape

                cost_error = cost_prime(prediction, target_value) * self.output_layer.activation_prime()
                layer_index = len(self.layers) - 1
                while layer_index >= 0:
                    layer = self.layers[layer_index]
                    layer.train(cost_error, activations[layer_index], learning_rate)

                    if layer_index == 0:
                        break
                    layer_index -= 1
                    layer = self.layers[layer_index]
                    next_layer = self.layers[layer_index + 1]
                    cost_error = np.dot(next_layer.weights.T, cost_error) * layer.activation_prime()

            correct_tests = self.evaluate(test_set)
            self.epoch_results.append(correct_tests)
            print(f"Epoch {epoch}: {correct_tests} / {len(test_set)} in {time.time() - start_timestamp} msec.")

    def evaluate(self, test_set):
        """Return the number of test inputs for which the neural network outputs the correct result. Note that the
        neural network's output is assumed to be the index of whichever neuron in the final layer has the highest
        activation."""

        correct_predictions = 0
        for input_data, target_value in test_set:
            if np.argmax(self.forward_propagation(input_data)) == np.argmax(target_value):
                correct_predictions += 1
        return correct_predictions
