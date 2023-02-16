import numpy as np


def one_hot(length, index):
    array = np.zeros((length, 1))
    array[index] = 1.0
    return array


def sigmoid(vector, out=None):
    if out is None:
        return 1.0 / (1.0 + np.exp(-vector))

    np.multiply(vector, -1.0, out=out)  # out = -vector
    np.exp(out, out=out)  # out = np.exp(-vector)
    np.add(out, 1.0, out=out)  # out = 1.0 + np.exp(-vector)
    identity = np.ones(vector.shape)
    np.divide(identity, out, out=out)  # out = 1.0 / 1.0 + np.exp(-vector)
    return out


def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


def cost_prime(prediction_value, target_value):
    """
    Return the vector of partial derivatives partial C_x partial a for the output activations. Both parameters and
    returned value are column vectors (a column vector is a matrix of n x 1).

    :param prediction_value the actual activation value produced by a layer,
    :param target_value expected target value from training set.
    :return cost function derivative, as a column vector.
    """
    assert prediction_value.shape[0] == target_value.shape[0]
    assert prediction_value.shape[1] == 1
    assert target_value.shape[1] == 1
    return prediction_value - target_value
