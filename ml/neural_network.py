import numpy as np
from numpy import random
import datetime
import time
import util


class NeuralLayer(object):
    """
    A layer (hidden or output) of the neural network. Theoretically a layer contains neurons but there is no such
    abstraction implemented. Instead a layer consist of floating point numbers stored on matrices; all matrices have
    the same number of rows (height), that is the layer size. Note that a matrix with a single column is also known as
    column vector.

    A layer has also an activation function (a) and layer output vector is a product and sum of matrices: A = a(WX + B),
    where X is the input vector. Activation function and its related derivative are configurable via constructor's
    activation name argument.

    Here are the matrices a layer is composed of:
    - W: weights matrix [layer_size x input_size]: neurons' weights for inputs vector,
    - B: biases vector [layer_size x 1]: neurons' bias,
    - Z: weighted inputs vector [layer_size x 1]: cache for computed weighted inputs: Z = WX + B,
    - A: activations vector [layer_size x 1]: cache for layer output vector: A = a(Z).

    Layer and input size are provided as arguments at layer creation. Input size the expected size of the input vector.
    It is not necessary - and usually isn't that input size to be the same as layer size; weights matrices dimension
    ensure that product weights x inputs (WX) produces a matrix of layer size height.
    """

    def __init__(self, layer_size, input_size, activation):
        """
        Initialize weights matrix and biases, weighted inputs and activations vectors; all matrices and vectors have the
        same number of rows. Weights matrix has the number of column equal with input size, whereas all vectors are
        matrices with a single column (aka. column vectors).

        Also initialize activation function and its derivative using provided activation name.

        :param layer_size: (int) the number of neurons from this layer,
        :param input_size: (int) the number of dimensions (features) input vector is expected to have,
        :param activation: (str) optional name of activation function used by this layer, possible None in which case
        default linear is used.
        """
        self.layer_size = layer_size
        self.input_size = input_size
        if not activation:
            activation = "linear"
        self.activation = activation

        # weights matrix of this layer inputs; matrix dimensions is layer_size x input_size
        self.weights = np.random.randn(layer_size, input_size)
        # neurons' biases vector is a matrix with dimensions layer_size x 1
        self.biases = np.random.rand(layer_size, 1)
        assert self.weights.shape[0] == self.biases.shape[0]

        # cache for last computed weighted inputs and activation values
        # updated by forward propagation and used for layer training
        self.weighted_inputs = np.zeros((layer_size, 1))
        self.activations = np.zeros((layer_size, 1))

        # initialize activation function and its related derivative
        match activation.lower():
            case "linear":
                self.activation_function = util.linear
                self.activation_function_prime = util.linear_prime

            case "relu":
                self.activation_function = util.relu
                self.activation_function_prime = util.relu_prime

            case "sigmoid" | _:
                self.activation_function = util.sigmoid
                self.activation_function_prime = util.sigmoid_prime

    def forward_propagation(self, input_data):
        """
        Propagates input data vector throw network's layers and return activation vector from output layer.

        :param input_data: (numpy.ndarray) network input data is a column vector (a matrix with a single column).
        :return: (numpy.ndarray) activation vector from output layer.
        """
        # np.dot operates also on matrices
        # input data parameter is a column vector (a matrix with a single column: n x 1)
        # for matrix dot product to be mathematically possible the number of columns of the first matrix
        # should be equal with the number rows of the second matrix
        # resulting matrix has the number of rows from the first matrix and the number of columns from the second
        # in our case: [m x n] dot [n x 1] -> [m x 1]

        assert input_data.shape == (self.input_size, 1)
        assert input_data.shape[0] == self.weights.shape[1]

        np.dot(self.weights, input_data, out=self.weighted_inputs)  # Z = W x X
        np.add(self.weighted_inputs, self.biases, out=self.weighted_inputs)  # Z = W x X + B
        self.activation_function(self.weighted_inputs, out=self.activations)  # A = sigma(Z)
        return self.activations

    def train(self, cost_error, activations, learning_rate):
        """
        Layer training method uses previous layer activations vector and current layer cost error to compute the cost
        function gradient for this layer. The gradient is then used, along with the learning rate, to adjust this
        layer parameters.

        This method assume current layer cost error is already calculated using back propagation algorithm.

        :param cost_error: (numpy.ndarray) cost error vector, computed with back propagation; matrix of layer_size x 1,
        :param activations: (numpy.ndarray) previous layer activations vector, matrix of input_size x 1,
        :param learning_rate: (float) learning rate.
        """
        assert cost_error.shape == (self.layer_size, 1)
        assert activations.shape == (self.input_size, 1)

        # use cost error to compute the cost function gradient with respect to weights
        weights_gradient = np.dot(cost_error, activations.T)
        # adjust weights gradient with learning rate
        np.multiply(weights_gradient, learning_rate, out=weights_gradient)
        # update this layer weights with cost gradient adjusted with learning rate
        np.subtract(self.weights, weights_gradient, out=self.weights)

        # cost error is the cost function gradient with respect to biases
        # adjust biases gradient with learning rate
        biases_gradient = np.multiply(cost_error, learning_rate)
        # update this layer biases with cost gradient adjusted with learning rate
        np.subtract(self.biases, biases_gradient, out=self.biases)

    def activation_prime(self):
        return self.activation_function_prime(self.weighted_inputs)

    def config(self):
        """
        Return this layer configuration. Current implementation has only two properties: layer size and activation
        function name, { layer_size, activation }.

        :return: (dict) this layer configuration.
        """
        return dict(size=self.layer_size, activation=self.activation)

    def dump(self, file):
        """
        Dump layer's parameters to binary file but do not close the file pointer. Every parameter (this is a float
        value) is saved in sequence to file on 4 bytes, accordingly IEEE 754 floating point standard. First are saved
        layer's weights then biases.

        This method leave the file pointer opened.

        :param file: (file pointer) file pointer opened for binary write.
        """

        weights = 0
        for index in np.ndindex(self.weights.shape):
            file.write(util.float2bytes(self.weights[index]))
            weights += 1

        biases = 0
        for index in np.ndindex(self.biases.shape):
            file.write(util.float2bytes(self.biases[index]))
            biases += 1

        print(f"dump weights:{weights}, biases:{biases}")


class NeuralNetwork(object):
    def __init__(self, input_size, *layers_meta):
        """
        Initialize the neural network instance. This constructor deals mainly with network's layers creation. For
        every layer there is a configuration tuple containing layer size and activation function name.

        :param input_size: (int) expected input vector dimension (or features number),
        :param layers_meta: (pointer) variable number of layer configuration tuples, one per layer.
        """
        self.input_size = input_size

        self.layers = []
        for layers_meta in layers_meta:
            layer_size, activation = layers_meta
            self.layers.append(NeuralLayer(layer_size, input_size, activation))
            # current layer size is the next layer input size
            input_size = layer_size

        self.output_layer = self.layers[-1]
        self.epoch_results = []
        self.accuracy = 0

    def forward_propagation(self, input_data):
        """
        Propagates input data vector throw network layers and returns network prediction. A layer output vector
        (aka activation vector) is sed as input vector for the next layer. Returns the output vector of the last layer,
        that is usually named output layer.

        :param input_data: (numpy.ndarray) input data as column vector (matrix with a single column).
        :return: (numpy.ndarray) network prediction vector, that is, the activation vector of the last layer.
        """
        assert input_data.shape == (self.input_size, 1)

        vector = input_data
        for layer in self.layers:
            vector = layer.forward_propagation(vector)
        return vector

    def train(self, train_set, test_set, epochs, learning_rate):
        """
        The goal of neural network training is to optimize network parameters for a minimal cost function. Parameters
        adjustment is performed using gradient descent algorithm. This algorithm uses cost function gradient with
        respect to network parameters; gradient is optimally computed using back propagation algorithm.

        This method deals with overall learning loop but the actual cost function gradient and parameters update is
        delegated to each layer. Anyway, layer cost error computation - part of back propagation algorithm, is
        performed on this method.

        Training process is repeated multiple time, as configured by epochs argument.

        Training and testing sets are sequences of (input data, target data) tuples where both input data and target
        value are column vectors (ndarray matrices with a single column).

        :param train_set: (list) training set is a sequence of tuple (input data, target data),
        :param test_set: (list) optional test set is a sequence of tuple (input data, target data), possible None,
        :param epochs: (int)  the number of training epochs (how may times training is repeated),
        :param learning_rate: (float) learning rate.
        """

        train_start_timestamp = time.time()
        best_correct_tests = 0

        for epoch in range(epochs):
            epoch_start_timestamp = time.time()
            random.shuffle(train_set)

            # activations references list stores previous activation vector for a given layer index
            # activations_references[i] is layers[i-1].activations; activations_references[0] is input data
            activations_references = [np.array([])] + [layer.activations for layer in self.layers[:-1]]

            for input_data, target_value in train_set:
                assert input_data.shape == (self.input_size, 1)
                activations_references[0] = input_data

                prediction = self.forward_propagation(input_data)
                assert prediction.shape == target_value.shape

                layer_index = len(self.layers) - 1
                # use back propagation algorithm to compute network (output layer) cost error
                # network cost error is then back propagated to previous layers
                cost_error = util.cost_prime(prediction, target_value) * self.output_layer.activation_prime()
                # print(f"cost_error: {cost_error}")
                while True:
                    # train current layer using its cost error
                    # activations references list is ordered so that it returns previous layer activations vector
                    current_layer = self.layers[layer_index]
                    current_layer.train(cost_error, activations_references[layer_index], learning_rate)

                    # step back and compute previous layer cost error using current layer cost error
                    if layer_index == 0:
                        break
                    layer_index -= 1

                    # here layer index was decremented and current layer is moved back
                    # so that next layer is actually the layer that was just trained
                    # compute cost error on the new current layer using cost error just used for training
                    current_layer = self.layers[layer_index]
                    next_layer = self.layers[layer_index + 1]
                    # print(f"next_layer.weights.T: {next_layer.weights.T}")
                    # print(f"cost_error: {cost_error}")
                    # print(f"current_layer.activation_prime(): {current_layer.activation_prime()}")
                    cost_error = np.dot(next_layer.weights.T, cost_error) * current_layer.activation_prime()

            if not test_set:
                print(f"Epoch {epoch} in {time.time() - epoch_start_timestamp} sec.")
                continue

            correct_tests = self.evaluate(test_set)
            if correct_tests > best_correct_tests:
                best_correct_tests = correct_tests
            self.epoch_results.append(correct_tests)
            print(f"Epoch {epoch}: {correct_tests} / {len(test_set)} in {time.time() - epoch_start_timestamp} sec.")

        print()
        print(f"Training finished in {time.time() - train_start_timestamp} sec.")
        if test_set:
            self.accuracy = best_correct_tests / (len(test_set) / 100)
            print(f"Best test evaluation {self.accuracy} %.")
        print()

    def evaluate(self, test_set):
        """
        Return the number of test inputs for which the neural network outputs the correct result. Note that the
        neural network's output is assumed to be the index of whichever neuron in the final layer has the highest
        activation.

        Network evaluation uses given test set. The test set is a sequence of tuple (input data, target data); input
        data is a vector (matrix with a single column) with dimension equal with input layer size. Expected target
        value is a vector with dimension equal with output layer size; it uses hot one encoding.

        :param test_set: (list) test set is a sequence of tuple (input data, target data).
        :return (int) the number of tests fulfilled with correct prediction.
        """

        correct_predictions = 0
        for input_data, target_value in test_set:
            if np.argmax(self.forward_propagation(input_data)) == np.argmax(target_value):
                correct_predictions += 1
        return correct_predictions

    def dump(self, network_name):
        config = dict(
            name=network_name,
            version="1.0",
            updated=datetime.date.today(),
            type="feedforward",
            accuracy=self.accuracy,
            input=self.input_size,
            layers=[layer.config() for layer in self.layers]
        )
        util.dump_config(config, network_name + ".yml")

        with open(network_name + ".data", "wb") as file:
            for layer in self.layers:
                layer.dump(file)
