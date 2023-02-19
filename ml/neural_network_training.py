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

nn = NeuralNetwork(784, (30, "sigmoid"), (10, "sigmoid"))
print("Start neural network's parameters optimization.")
nn.train(train_data, test_data, 30, 0.3)
print(nn.epoch_results)

# comment out network dump since currently saved accuracy is around its maximum - 95.1 %
# nn.dump("demo")

result_x = range(0, len(nn.epoch_results))
result_y = nn.epoch_results

spline_fun = interp.make_interp_spline(result_x, result_y)
spline_x = np.linspace(0, len(result_x), 500)
spline_y = spline_fun(spline_x)

graph.plot(result_x, result_y, 'o')
graph.plot(spline_x, spline_y)
graph.show()
