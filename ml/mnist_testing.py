import pickle
import numpy as np
import json
import util


class NeuralNetwork(object):
    def __init__(self, json_file):
        data = json.load(json_file)
        self.biases = np.asarray(data["biases"], dtype=object)
        self.weights = np.asarray(data["weights"], dtype=object)

    def feed_forward(self, data_vector):
        for layer_weights_matrix, layer_biases_array in zip(self.weights, self.biases):
            data_vector = util.sigmoid(np.dot(layer_weights_matrix, data_vector) + layer_biases_array)
        return data_vector


def load_test_set():
    with open("mnist.pkl", "rb") as file:
        data_set = pickle.load(file, encoding="latin1")

    test_set = data_set[2]
    test_images = [np.reshape(image, (784, 1)) for image in test_set[0]]
    return test_images, test_set[1]


def main():
    test_set = load_test_set()
    image_vectors = test_set[0]
    predicted_values = test_set[1]

    with open("network.json", "r", encoding="utf-8") as file:
        neural_network = NeuralNetwork(file)

    total = 0
    correct = 0
    fails = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for image_vector, predicted_value in zip(image_vectors, predicted_values):
        output_vector = neural_network.feed_forward(image_vector)
        # print(f"expected:{predicted_value} actual:{np.argmax(output_vector)}")
        total += 1
        if predicted_value == np.argmax(output_vector):
            correct += 1
        else:
            fails[predicted_value] = fails[predicted_value] + 1

    print(f"total:{total} correct:{correct}")
    print(fails)


main()
