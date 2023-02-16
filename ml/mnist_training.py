import pickle
import random
import json
import numpy as np
import matplotlib.pylab as graph
import scipy.interpolate as interp
import util

with open("mnist.pkl", "rb") as json_file:
    data_set = pickle.load(json_file, encoding="latin1")

train_set = data_set[0]
train_inputs = [np.reshape(image, (784, 1)) for image in train_set[0]]
train_targets = [util.one_hot(10, label) for label in train_set[1]]
train_data = list(zip(train_inputs, train_targets))

test_set = data_set[2]
test_inputs = [np.reshape(image, (784, 1)) for image in test_set[0]]
test_targets = [util.one_hot(10, label) for label in test_set[1]]
test_data = list(zip(test_inputs, test_targets))


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.epoch_results = []

        # - + + -> layer size (all layers but not input layer)
        # | | |
        # + + - -> previous layer size (all layers but not output layer)
        self.biases = [np.random.randn(layer_size, 1) for layer_size in sizes[1:]]
        self.weights = [np.random.randn(layer_size, previous_layer_size) for layer_size, previous_layer_size in
                        zip(sizes[1:], sizes[:-1])]

    def feed_forward(self, a):
        """
        Return the output of the network if 'a' is the input.
        :param a:
        :return:
        """
        for b, w in zip(self.biases, self.weights):
            a = util.sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descent(self, training_data, testing_data, epochs, batch_size, learning_rate):
        tests_count = len(testing_data)
        training_data_length = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            train_batches = [training_data[batch_index:batch_index + batch_size] for batch_index in
                             range(0, training_data_length, batch_size)]

            for train_batch in train_batches:
                self.update_batch(train_batch, learning_rate)

            correct_tests = self.evaluate(testing_data)
            self.epoch_results.append(correct_tests)
            print(f"Epoch {epoch}: {correct_tests} / {tests_count}")

    def update_batch(self, train_batch, learning_rate):
        """Update the network's parameters - weights and biases, by applying gradient descent using backpropagation to
        a single batch from the train set. The train batch is a list of tuples (input_data, target_value), both being
        column vectors (a column vector is a matrix n x 1). Input data is injected into layer neurons whereas target
        value is the expected output.

        :param train_batch a chunk from train set, with the same structure,
        :param learning_rate learning rate.
        """

        # the sum per batch of the weights and biases gradient
        weights_batch_gradient = [np.zeros(weight.shape) for weight in self.weights]
        biases_batch_gradient = [np.zeros(bias.shape) for bias in self.biases]

        for input_data, target_value in train_batch:
            weights_gradient, biases_gradient = self.back_propagation(input_data, target_value)
            weights_batch_gradient = [batch_gradient + gradient for batch_gradient, gradient in
                                      zip(weights_batch_gradient, weights_gradient)]
            biases_batch_gradient = [batch_gradient + gradient for batch_gradient, gradient in
                                     zip(biases_batch_gradient, biases_gradient)]

        learning_rate = learning_rate / len(train_batch)
        self.weights = [weight - learning_rate * gradient for weight, gradient in
                        zip(self.weights, weights_batch_gradient)]
        self.biases = [bias - learning_rate * gradient for bias, gradient in
                       zip(self.biases, biases_batch_gradient)]

    def back_propagation(self, input_data, target_value):
        """Return a tuple ``(del_b , del_w)`` representing the
        gradient for the cost function C_x. ``del_b `` and
        ``del_w `` are layer -by-layer lists of numpy arrays , similar
        to ``self.biases `` and ``self.weights ``."""

        weights_gradient = [np.zeros(layer_weights.shape) for layer_weights in self.weights]
        biases_gradient = [np.zeros(layer_biases.shape) for layer_biases in self.biases]

        # feedforward
        activation = input_data
        activations = [input_data]  # list to store all the activations , layer by layer
        weighted_inputs = []  # list to store all the z vectors , layer by layer
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            weighted_input = np.dot(layer_weights, activation) + layer_biases
            weighted_inputs.append(weighted_input)
            activation = util.sigmoid(weighted_input)
            activations.append(activation)

        # back-propagation algorithm for computing cost function gradient
        # it starts form output layer, then backward throw hidden layers

        cost_error = util.cost_prime(activations[-1], target_value) * util.sigmoid_prime(weighted_inputs[-1])
        weights_gradient[-1] = np.dot(cost_error, activations[-2].T)
        biases_gradient[-1] = cost_error

        # Note that the variable l in the loop below is used a little

        # differently to the notation in Chapter 2 of the book. Here ,
        # l = 1 means the last layer of neurons , l = 2 is the
        # second -last layer , and so on. It's a renumbering of the
        # scheme in the book , used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            cost_error = np.dot(self.weights[-layer + 1].T, cost_error) * util.sigmoid_prime(weighted_inputs[-layer])
            weights_gradient[-layer] = np.dot(cost_error, activations[-layer - 1].T)
            biases_gradient[-layer] = cost_error

        return weights_gradient, biases_gradient

    def evaluate(self, testing_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        testing_results = [(self.feed_forward(x), y) for (x, y) in testing_data]
        return sum(int(np.argmax(x) == np.argmax(y)) for (x, y) in testing_results)

    def save(self, file):
        biases = [ndarray.tolist() for ndarray in self.biases]
        weights = [ndarray.tolist() for ndarray in self.weights]
        data = {'num_layers': self.num_layers, 'sizes': self.sizes, 'biases': biases, 'weights': weights}
        file.write(json.dumps(data))


neural_network = NeuralNetwork([784, 30, 10])
print(neural_network)

EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 0.3
neural_network.gradient_descent(train_data, test_data, EPOCHS, BATCH_SIZE, LEARNING_RATE)

with open("network.json", "tw") as json_file:
    neural_network.save(json_file)

print(neural_network.epoch_results)

result_x = range(0, len(neural_network.epoch_results))
result_y = neural_network.epoch_results

spline_fun = interp.make_interp_spline(result_x, result_y)
spline_x = np.linspace(0, len(result_x), 500)
spline_y = spline_fun(spline_x)

graph.plot(result_x, result_y, 'o')
graph.plot(spline_x, spline_y)
graph.show()
