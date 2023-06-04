import numpy as np
import matplotlib.pyplot as plot
from neural_network import NeuralNetwork

x = np.linspace(0, 10, 100).reshape(100, 1)
y = x ** 2
print(f"x shape: {x.shape}, y shape: {y.shape}")

train_set = []
for i in range(100):
    train_set.append((np.array([x[i]]), np.array([y[i]])))

nn = NeuralNetwork(1, (10, "relu"), (1, "linear"))
nn.train(train_set, None, 10, 0.01)

value = 64
print(x[value])
print(y[value])
prediction = nn.forward_propagation(np.array([x[value]]))
print(prediction)

fig = plot.figure()
# plot.plot(x, y, 'r')
plot.scatter(x, y, 1)

nn_x = np.linspace(0, 10, 100)
nn_y = []
for i in nn_x:
    input_vector = np.array([[i]])
    prediction_vector = nn.forward_propagation(input_vector)
    nn_y.append(prediction_vector[0])
plot.plot(nn_x, nn_y, 'g')

plot.show()
