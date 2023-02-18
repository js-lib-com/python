import pickle
import numpy as np
from neural_network import NeuralNetwork
from util import one_hot
import matplotlib.pylab as graph
import scipy.interpolate as interp

with open("mnist.pkl", "rb") as json_file:
    data_set = pickle.load(json_file, encoding="latin1")

train_set = data_set[0]
train_inputs = [np.reshape(image, (784, 1)) for image in train_set[0]]
train_targets = [one_hot(10, label) for label in train_set[1]]
train_data = list(zip(train_inputs, train_targets))

test_set = data_set[2]
test_inputs = [np.reshape(image, (784, 1)) for image in test_set[0]]
test_targets = [one_hot(10, label) for label in test_set[1]]
test_data = list(zip(test_inputs, test_targets))

neural_network = NeuralNetwork(784, (30, "sigmoid"), (10, "sigmoid"))
print(neural_network)

neural_network.train(train_data, test_data, 30, 0.3)

print(neural_network.epoch_results)

neural_network.dump("xdemo")

result_x = range(0, len(neural_network.epoch_results))
result_y = neural_network.epoch_results

spline_fun = interp.make_interp_spline(result_x, result_y)
spline_x = np.linspace(0, len(result_x), 500)
spline_y = spline_fun(spline_x)

graph.plot(result_x, result_y, 'o')
graph.plot(spline_x, spline_y)
graph.show()

"""
config = util.config("demo.yml")
print(config.nn.layers[0].size)
print(config.nn.layers[0].activation)
for layer in config.nn.layers:
    print(f"size: {layer.size}, activation: {layer.activation}")

pi = 3.14159
bytes_ = util.float2bytes(pi)
print(bytes_)  # prints: bytearray(b'@I\xfb\x1f')

bytes_ = b'@I\xfb\x1f'
pi = util.bytes2float(bytes_)
print(pi)  # prints: 3.141590118408203
"""
