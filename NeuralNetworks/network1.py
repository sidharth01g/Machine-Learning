from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
import pyprind

np.set_printoptions(threshold=np.nan)


class Network(object):

    def __init__(self, node_counts, activation_function_hidden,
                 activation_derivative_function_hidden,
                 weights_init_factor=0.01):

        assert(type(node_counts) is list)
        assert(type(weights_init_factor) is float)
        # np.random.seed(1)
        self.activation_function_hidden = activation_function_hidden
        self.activation_derivative_function_hidden = (
            activation_derivative_function_hidden
        )

        self.weights = {}
        self.biases = {}

        for i in range(1, len(node_counts)):
            self.weights[i] = (
                weights_init_factor
                * np.random.randn(node_counts[i], node_counts[i - 1])
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

    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def forward_propagate(self, X):
        self.Z = {0: None}
        self.A = {0: X}
        for layer in self.weights:
            self.Z[layer] = (
                np.dot(self.weights[layer], self.A[layer - 1])
                + self.biases[layer]
            )

            if layer == self.L:
                self.A[layer] = Network.sigmoid(self.Z[layer])
            else:
                # self.A[layer] = np.tanh(self.Z[layer])
                self.A[layer] = self.activation_function_hidden(self.Z[layer])
        # print(self.Z[3])
        # exit()
        return self.A[self.L]

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
        self.dA = {
            self.L: (-Y / self.A[self.L]) + (1 - Y) / (1 - self.A[self.L])
        }
        self.dZ = {self.L: self.A[self.L] - Y}
        """
        self.dZ = {
            self.L: self.dA[self.L] * self.A[self.L] * (1 - self.A[self.L])
        }
        """
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
            self.dA[layer] = np.dot(
                self.weights[layer + 1].T, self.dZ[layer + 1]
            )
            self.dZ[layer] = np.multiply(
                self.dA[layer],
                self.activation_derivative_function_hidden(self.Z[layer])
            )
            self.dW[layer] = (
                (1.0 / m)
                * np.dot(self.dZ[layer], self.A[layer - 1].T)
            )
            self.db[layer] = (
                (1.0 / m)
                * np.sum(self.dZ[layer], axis=1, keepdims=True)
            )

    def update_parameters(self, learning_rate):
        assert(type(learning_rate) is float)
        for layer in range(1, self.L + 1):
            self.weights[layer] = (
                self.weights[layer] - learning_rate * self.dW[layer]
            )
            self.biases[layer] = (
                self.biases[layer] - learning_rate * self.db[layer]
            )
        print('\n', self.weights[self.L], learning_rate * self.dW[self.L])

    def run_gradient_descent(self, X, Y, learning_rate, epochs):
        assert(type(learning_rate) is float)
        assert(type(epochs) is int)
        block_character = bytes((219,)).decode('cp437')
        progress_bar = pyprind.ProgBar(
            epochs, monitor=True, title='Training the Neural Network..',
            bar_char=block_character
        )

        costs = []
        for i in range(epochs):
            self.forward_propagate(X)
            costs.append(self.get_cost(Y))
            self.back_propagate(Y)
            self.update_parameters(learning_rate)
            progress_bar.update()
        return costs


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

    # Class 0: digits[0], Class 1: digits[1]
    digits = [0, 6]
    # Only 2 classes allowed (binary classification between 2 digits)
    assert(len(digits) == 2)
    temp_0 = X[y == digits[0]].T
    temp_1 = X[y == digits[1]].T
    temp_y_0 = y[y == digits[0]]
    temp_y_1 = y[y == digits[1]]
    temp_y_0 = temp_y_0.reshape(1, temp_y_0.shape[0])
    temp_y_1 = temp_y_1.reshape(1, temp_y_1.shape[0])

    X = np.hstack((temp_0, temp_1))
    y = np.hstack((temp_y_0, temp_y_1))
    permutation = np.random.permutation(X.shape[1])
    X = X[:, permutation]
    y = y[:, permutation]

    X = X / np.amax(X)

    # Map values of y to 1 and 0 (output layer has sigmoidal activation)
    # i.e output is restricted to [0, 1]
    mapping = {digits[0]: 0, digits[1]: 1}
    y_copy = np.copy(y)
    for k, v in mapping.items():
        y_copy[y == k] = v

    y = y_copy

    train_size = int(train_ratio * X.shape[1])

    x_train = X[:, :train_size]
    y_train = y[:, :train_size]

    x_test = X[:, train_size:]
    y_test = y[:, train_size:]
    # show_samples(x_train, y_train, [1, 2, 3])
    return (x_train, x_test, y_train, y_test)


def show_samples(x, y, indices_list):
    for index in indices_list:
        digit_serial = x[:, index]
        digit_reshaped = digit_serial.reshape(28, 28)
        # pp.pprint(digit_reshaped)
        plt.figure()
        plt.imshow(digit_reshaped, cmap='Greys', interpolation='nearest')
        plt.title(str(y[0, index]))
    plt.show()


def get_activation_functions(function_name):

    def sigmoid(Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def sigmoid_derivative(Z):
        return sigmoid(Z) * (1 - sigmoid(Z))

    def relu(Z):
        return np.maximum(0, Z)

    def relu_derivative(Z):
        deriv = np.ones(Z.shape)
        deriv[Z <= 0] = 0.0
        # print(deriv)
        return deriv

    mapping = {
        'tanh': (np.tanh, lambda z: 1 - np.power(np.tanh(z), 2)),
        'sigmoid': (sigmoid, sigmoid_derivative),
        'relu': (relu, relu_derivative)
    }
    return mapping[function_name]


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

    # Initialize network
    node_counts = [x_train.shape[0], 10, 5, y_train.shape[0]]
    (activation_function_hidden, activation_derivative_function_hidden) = (
        get_activation_functions('sigmoid')
    )
    net = Network(
        node_counts, activation_function_hidden,
        activation_derivative_function_hidden, weights_init_factor=0.1)
    heading('Neural Network parameters')
    for i in range(1, len(node_counts)):
        print(
            'Layer: %s, Weights: %s, Biases: %s'
            % (i, net.weights[i].shape, net.biases[i].shape)
        )

    heading('Forward propagation')
    Y_hat = net.forward_propagate(x_train)
    print('Y_hat.shape', Y_hat.shape)

    heading('Test cost computation')
    print('Cost: ', net.get_cost(y_train))

    """
    heading('Back propagation')
    net.back_propagate(y_train)
    net.update_parameters(learning_rate=1.0)

    del(net)
    # net = Network(node_counts, weights_init_factor=0.1)
    net = Network(
        node_counts, activation_function_hidden,
        activation_derivative_function_hidden, weights_init_factor=0.01)

    """
    heading('Gradient descent')
    learning_rate = 1.0
    epochs = 100
    costs = net.run_gradient_descent(
        X=x_train, Y=y_train, learning_rate=learning_rate, epochs=epochs)
    print('Costs: ', costs)

    heading('Testing')
    Y_predict = net.forward_propagate(x_test)
    # print(Y_predict)
    m_test = y_test.shape[1]
    Y_thresh = Y_predict > 0.5
    y_test_transormed = (y_test == 1)
    score = np.sum(Y_thresh == y_test_transormed) / m_test
    print('Score: %s percent' % (score * 100))


if __name__ == '__main__':
    test()
