from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
import pyprind

np.set_printoptions(threshold=np.nan)


class Network(object):

    def __init__(self, node_counts, weights_init_factor=0.01):
        assert(type(node_counts) is list)
        self.weights = {}
        self.biases = {}
        for i in range(1, len(node_counts)):
            self.weights[i] = (
                weights_init_factor
                * np.random.rand(node_counts[i], node_counts[i - 1])
            )
            self.biases[i] = (
                np.zeros((node_counts[i], 1), dtype=float)
            )

        self.weight_gradients = None
        self.bias_gradients = None

    @property
    def L(self):
        # Biases and weights should tally (to the number of layers)
        assert(len(self.weights) == len(self.biases))
        return len(self.weights)

    def forward_propagate(self, X):
        self.Z = {0: None}
        self.A = {0: X}
        for layer in self.weights:
            print('\nLayer: ', layer)
            self.Z[layer] = (
                np.dot(self.weights[layer], self.A[layer - 1])
                + self.biases[layer]
            )
            self.A[layer] = np.tanh(self.Z[layer])
            print('Z: ', self.Z[layer].shape)
            print('A: ', self.A[layer].shape)

    def get_cost(self, Y):
        # m: number of training examples
        m = Y.shape[1]

        # predicted output = activation at the output (deepest) layer
        Y_hat = self.A[self.L]
        cross_entropy_loss = (
            Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
        )
        cost = (-1.0 / m) * np.sum(cross_entropy_loss)
        cost = np.squeeze(cost)

        return cost

    def back_propagate(self, Y):
        m = Y.shape[1]

        # Backpropagation at the output layer L
        self.dZ = {self.L: self.A[self.L] - Y}
        self.dW = {
            self.L: (
                (1.0 / m)
                * np.dot(self.dZ[self.L], self.A[self.L - 1].T)
            )
        }
        self.db = {
            self.L: (
                (1.0 / m)
                * np.sum(self.dZ[self.L], axis=1, keepdims=True)
            )
        }

        # Backpropagation at other layers (L-1, L-2, ... , 1)
        for layer in range(self.L - 1, 0, -1):
            self.dZ[layer] = (
                (np.dot(self.weights[layer + 1].T, self.dZ[layer + 1]))
                * (1.0 - np.power(self.A[layer], 2))
            )

            self.dW[layer] = (
                (1.0 / m) * np.dot(self.dZ[layer], self.A[layer - 1].T)
            )

            self.db[layer] = (
                (1.0 / m) * np.sum(self.dZ[layer], axis=1, keepdims=True)
            )

    def update_parameters(self, learning_rate):
        for layer in range(1, self.L):
            self.weights[layer] = (
                self.weights[layer] - learning_rate * self.dW[layer]
            )
            self.biases[layer] = (
                self.biases[layer] - learning_rate * self.db[layer]
            )
            print(self.weights[layer].shape, self.biases[layer].shape)
    """
    def run_gradient_descent(self, X, Y, learning_rate, epochs):
        for i in range(epochs):
            forward_propagate(self, X)
    """


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
    heading('Neural Network parameters')
    for i in range(1, len(node_counts)):
        print(
            'Layer: %s, Weights: %s, Biases: %s'
            % (i, net.weights[i].shape, net.biases[i].shape)
        )

    heading('Forward propagation')
    net.forward_propagate(x_train)

    heading('Test cost computation')
    print('Cost: ', net.get_cost(y_train))

    heading('Back propagation')
    net.back_propagate(y_train)
    net.update_parameters(learning_rate=0.1)


if __name__ == '__main__':
    test()
