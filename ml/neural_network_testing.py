import pickle
import time
import numpy as np
import matplotlib.pyplot as plot
from munch import DefaultMunch

import neural_network_worker
import util

print("Start network testing.")
config = DefaultMunch.fromDict(util.load_config("demo.yml"))
inn = neural_network_worker.NeuralNetworkWorker(config, "demo.data")

with open("mnist.pkl", "rb") as file:
    data_set = pickle.load(file, encoding="latin1")

test_set = data_set[2]
test_images = test_set[0]
test_values = test_set[1]
print("Tests set loaded.")

start_timestamp = time.time()
correct_predictions = 0
results = np.zeros((10, 10), dtype=int)
for test_image, target_value in zip(test_images, test_values):
    input_vector = test_image.reshape(784, 1)
    output_vector = inn.forward_propagation(input_vector)

    prediction = np.argmax(output_vector)
    if prediction == target_value:
        correct_predictions += 1
    results[target_value][prediction] += 1

print()
print(f"Testing finished in {time.time() - start_timestamp} sec.")
print(f"Total tests: {len(test_images)}")
print(f"Correct predictions: {correct_predictions}")
print(f"Network accuracy: {correct_predictions / (len(test_images) / 100)} %.")
print()
print(results)

X, Y, Z = [], [], []
for index in np.ndindex(results.shape):
    X.append(index[0])
    Y.append(index[1])
    Z.append(min(30, results[index]))

colors = ["black", "red", "green", "blue", "gold", "brown", "purple", "magenta", "cyan", "slategray"]
C = [colors[y % 10] for y in Y]

figure = plot.figure()
canvas = figure.canvas

subplot = figure.add_subplot(projection='3d')
# subplot.axis("off")
subplot.plot(range(0, 10), [8 for x in range(0, 10)], [Z[8 + 10 * x] for x in range(0, 10)])
subplot.scatter(X, Y, Z, c=C)
plot.show()
