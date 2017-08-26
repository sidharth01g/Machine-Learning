from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
import pyprind

np.set_printoptions(threshold=np.nan)


class Network(object):

    def __init__(self, node_counts, weights_init_factor=0.01):
        assert(type(node_counts) is list)
        self.weights = []
        self.biases = []
        for i in range(1, len(node_counts)):
            self.weights.append(
                weights_init_factor
                * np.random.rand(node_counts[i], node_counts[i - 1])
            )
            self.biases.append(
                np.zeros((node_counts[i], 1), dtype=float)
            )

def load_mnist(train_ratio, data_dir=None):
    # Perform import specific to test() method
    import os
    import sys
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    sys.path.insert(0, parent_dir)
    import numpy as np

    from utils.common import load_mnist_dataset

    # Prepare training data

    if not data_dir:
        data_dir = os.path.join(parent_dir, 'downloads', 'MNIST')
    print('Dataset directory: ', data_dir)
    (X, y) = load_mnist_dataset(data_dir)
    temp_0 = X[y == 0].T
    temp_1 = X[y == 1].T
    temp_y_0 = y[y == 0]
    temp_y_1 = y[y == 1]
    temp_y_0 = temp_y_0.reshape(1, temp_y_0.shape[0])
    temp_y_1 = temp_y_1.reshape(1, temp_y_1.shape[0])

    X = np.hstack((temp_0, temp_1))
    y = np.hstack((temp_y_0, temp_y_1))
    permutation = np.random.permutation(X.shape[1])
    X = X[:, permutation]
    y = y[:, permutation]

    X = X / np.amax(X)

    train_size = int(train_ratio * X.shape[1])

    x_train = X[:, :train_size]
    y_train = y[:, :train_size]

    x_test = X[:, train_size:]
    y_test = y[:, train_size:]
    return (x_train, x_test, y_train, y_test)



def test():
    # Import methods for running test()
    import os
    import sys

    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    sys.path.insert(0, parent_dir)
    from utils.common import heading

    # Fetch dataset
    heading('Data preparation')
    train_ratio = 0.7
    (x_train, x_test, y_train, y_test) = load_mnist(train_ratio)
    print('train_ratio', train_ratio)
    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)
    print('x_test: ', x_test.shape)
    print('y_test: ', y_test.shape)

    # Intitialize network
    node_counts = [x_train.shape[0], 3, 5, y_train.shape[0]]
    net = Network(node_counts)
    heading('Nerural Network parameters')
    for i in range(len(node_counts) - 1):
        print(
            'Layer: %s, Weights: %s, Biases: %s'
            % (i, net.weights[i].shape, net.biases[i].shape)
        )


if __name__ == '__main__':
    test()
