import numpy as np
import yaml
from yaml.loader import SafeLoader
import ruamel.yaml
import struct


def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=SafeLoader)


def dump_config(config, file_path):
    ruamel_yaml = ruamel.yaml.YAML()
    ruamel_yaml.preserve_quotes = True
    ruamel_yaml.indent(mapping=2, sequence=4, offset=2)
    ruamel_yaml.width = 120
    ruamel_yaml.allow_unicode = True

    with open(file_path, "w") as file:
        ruamel_yaml.dump(config, file)


def float2bytes(f):
    # Convert float to 32-bit binary string
    bits = struct.pack('!f', f)
    # Return binary string as 4 bytes
    return bytearray(bits)


def bytes2float(bytes_):
    # Convert 4 bytes to binary string
    bits = bytes(bytes_)
    # Unpack binary string as 32-bit float value
    f = struct.unpack('!f', bits)[0]
    return f


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


def relu(vector, out=None):
    if out is None:
        return np.maximum(0.0, vector)

    np.maximum(0.0, vector, out=out)
    return out


def relu_prime(x):
    return x > 0


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
