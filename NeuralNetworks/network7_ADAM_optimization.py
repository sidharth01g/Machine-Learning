import copy
from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
import pyprind
import warnings

np.set_printoptions(threshold=np.nan)


class Network(object):

    def __init__(self, node_counts, activation_function_hidden,
                 activation_derivative_function_hidden, lambd=None,
                 keep_probability=None):

        """Initialize neural network

        Arguments:
        ==========
        node_counts - list. Number of neurons in each layer
        activation_function_hidden - string. Hidden layer activation function
        activation_derivative_function_hidden - string
        lambd - float. L2 regularization parameter
        keep_probability - float. Dropout keep_probability
        alpha - float. Momentum gradient descent hyperparameter
        beta - float. Momentum gradient descent hyperparameter
        """

        assert(type(node_counts) is list)
        assert(lambd is None or type(lambd) is float)
        assert(
            keep_probability is None or (
                type(keep_probability) is float
                and 0.0 <= keep_probability <= 1.0)
        )

        self.activation_function_hidden = activation_function_hidden
        self.activation_derivative_function_hidden = (
            activation_derivative_function_hidden
        )

        # L2 regularization parameter
        self.lambd = lambd

        # Dropout regularization parameter: probability of keeping node active
        self.keep_probability = keep_probability

        self.weights = {}
        self.biases = {}

        for i in range(1, len(node_counts)):
            # He et.al. initialization: works well with ReLU activation
            weights_init_factor = np.sqrt(1 / node_counts[i - 1])
            self.weights[i] = (
                weights_init_factor
                * np.random.randn(node_counts[i], node_counts[i - 1])
            )
            self.biases[i] = (
                np.zeros((node_counts[i], 1), dtype=float)
            )

        self.weight_gradients = None
        self.bias_gradients = None
        self.VdW = None
        self.Vdb = None
        self.SdW = None
        self.Sdb = None
        self.t = 0

    @property
    def L(self):
        # Biases and weights should tally (to the number of layers)
        assert(len(self.weights) == len(self.biases))
        return len(self.weights)

    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def predict(self, X):
        return self.forward_propagate(X, train=False)

    def forward_propagate(self, X, train=True):
        self.Z = {0: None}
        self.A = {0: X}
        # Dropout mask
        self.D = {0: None}
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

                if self.keep_probability and train:
                    # Create and apply dropout mask
                    self.D[layer] = np.random.rand(
                        self.A[layer].shape[0], self.A[layer].shape[1]
                    )
                    self.D[layer] = self.D[layer] < self.keep_probability
                    self.A[layer] = (
                        (1.0 / self.keep_probability)
                        * self.A[layer] * self.D[layer]
                    )

        # print(self.Z[3])
        # exit()
        return self.A[self.L]

    def get_cost(self, Y):
        # m: number of training examples
        m = Y.shape[1]

        # predicted output = activation at the output (deepest) layer
        Y_hat = self.A[self.L]
        cross_entropy_loss = (
            (-1.0 / m)
            * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        )
        if self.lambd:
            l2_regularization_cost = (
                0.5 * (self.lambd / m)
                * np.sum(np.sum(np.power(w, 2)) for w in self.weights)
            )
        else:
            l2_regularization_cost = 0
        cost = cross_entropy_loss + l2_regularization_cost
        cost = np.squeeze(cost)

        return cost

    def back_propagate(self, Y):
        m = Y.shape[1]

        # Backpropagation at the output layer L
        warnings.filterwarnings('error')
        try:
            self.dA = {
                self.L: (-Y / self.A[self.L]) + (1 - Y) / (1 - self.A[self.L])
            }
            self.dZ = {self.L: self.A[self.L] - Y}
        except Warning:
            #print(warning)
            # print(self.A[self.L] == 0)
            print(0 >= self.A[self.L] >= 1)
            exit('<<<<<<<<Exit')
        warnings.filterwarnings('default')

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
        # Add L2 regularization term
        if self.lambd:
            self.dW[self.L] += (self.lambd / m) * self.weights[self.L]

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

            # Dropout
            if self.keep_probability:
                self.dA[layer] = (
                    (1.0 / self.keep_probability)
                    * self.dA[layer] * self.D[layer]
                )

            self.dZ[layer] = np.multiply(
                self.dA[layer],
                self.activation_derivative_function_hidden(self.Z[layer])
            )
            self.dW[layer] = (
                (1.0 / m)
                * np.dot(self.dZ[layer], self.A[layer - 1].T)
            )
            # Add L2 regularization term
            if self.lambd:
                self.dW[layer] += (self.lambd / m) * self.weights[layer]

            self.db[layer] = (
                (1.0 / m)
                * np.sum(self.dZ[layer], axis=1, keepdims=True)
            )

    def update_parameters(self, learning_rate, alpha, beta1=0.9, beta2=0.99,
                          eps=1e-3):
        assert(learning_rate is None or type(learning_rate) is float)
        assert(alpha is None or type(alpha) is float)
        assert(beta1 is None or type(beta1) is float)
        assert(beta2 is None or type(beta2) is float)
        assert(learning_rate or (alpha and beta1 and beta2))
        assert(0.0 <= beta1 <= 1.0 and 0.0 <= beta2 <= 1.0)

        if learning_rate:
            for layer in range(1, self.L + 1):
                self.weights[layer] = (
                    self.weights[layer] - learning_rate * self.dW[layer]
                )
                self.biases[layer] = (
                    self.biases[layer] - learning_rate * self.db[layer]
                )
        else:
            # Set moving average to be equal to the first gradient
            if not self.VdW and not self.Vdb and not self.SdW and not self.Sdb:
                """
                self.VdW = copy.deepcopy(self.dW)
                self.Vdb = copy.deepcopy(self.db)
                self.SdW = {l: np.power(w, 2) for l, w in self.dW.items()}
                self.Sdb = {l: np.power(b, 2) for l, b in self.db.items()}
                """
                self.VdW = {l: np.zeros(dw.shape) for l, dw in self.dW.items()}
                self.Vdb = {l: np.zeros(db.shape) for l, db in self.db.items()}
                self.SdW = {l: np.zeros(dw.shape) for l, dw in self.dW.items()}
                self.Sdb = {l: np.zeros(db.shape) for l, db in self.db.items()}

            for layer in range(1, self.L + 1):
                self.VdW[layer] = (
                    beta1 * self.VdW[layer]
                    + (1 - beta1) * self.dW[layer]
                )
                self.Vdb[layer] = (
                    beta1 * self.Vdb[layer]
                    + (1 - beta1) * self.db[layer]
                )
                self.SdW[layer] = (
                    beta2 * self.SdW[layer]
                    + (1 - beta2) * np.power(self.dW[layer], 2)
                )
                self.SdW[layer] = (
                    beta2 * self.Sdb[layer]
                    + (1 - beta2) * np.power(self.db[layer], 2)
                )

                VdW_corrected = self.VdW[layer] / (1 - np.power(beta1, self.t))
                Vdb_corrected = self.Vdb[layer] / (1 - np.power(beta1, self.t))
                SdW_corrected = self.SdW[layer] / (1 - np.power(beta2, self.t))
                Sdb_corrected = self.Sdb[layer] / (1 - np.power(beta2, self.t))

                self.weights[layer] = (
                    self.weights[layer]
                    - alpha * VdW_corrected / np.sqrt(SdW_corrected + eps)
                )
                self.biases[layer] = (
                    self.biases[layer]
                    - alpha * Vdb_corrected / np.sqrt(Sdb_corrected + eps)
                )
            print(self.t, np.power(beta1, self.t), np.power(beta2, self.t))

    def run_gradient_descent(self, X, Y, epochs, learning_rate=None,
                             alpha=None, beta=None):
        assert(learning_rate is None or type(learning_rate) is float)
        assert(alpha is None or type(alpha) is float)
        assert(beta is None or type(beta) is float)
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
            self.t += 1
            self.update_parameters(learning_rate, alpha, beta1=0.9, beta2=0.99,
                                   eps=0.01)
            progress_bar.update()
        return costs


def load_mnist(train_ratio, digits, data_dir=None):
    assert(type(train_ratio) is float)
    assert(type(digits) is list and len(digits) == 2)
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
    # e.g. digits = [7, 8]
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


def show_samples(x, y, indices_list, mapping):
    for index in indices_list:
        digit_serial = x[:, index]
        digit_reshaped = digit_serial.reshape(28, 28)
        # pp.pprint(digit_reshaped)
        plt.figure()
        plt.imshow(digit_reshaped, cmap='Greys', interpolation='nearest')
        plt.title('Prediction: ' + str(mapping[y[0, index]]))


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


def plot_cost(costs, parameters):
    plt.figure()
    plt.title('Learning parameters=' + str(parameters))
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.plot(costs)
    plt.legend(loc='best')


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
    digits = [3, 9]
    (x_train, x_test, y_train, y_test) = load_mnist(train_ratio, digits)
    print('train_ratio', train_ratio)
    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)
    print('x_test: ', x_test.shape)
    print('y_test: ', y_test.shape)

    # Initialize network
    node_counts = [x_train.shape[0], 20, 10, 10, y_train.shape[0]]
    (activation_function_hidden, activation_derivative_function_hidden) = (
        get_activation_functions('relu')
    )
    """
    net = Network(
        node_counts, activation_function_hidden,
        activation_derivative_function_hidden, lambd=1.5, keep_probability=0.7)
    """
    # L2 regularization parameter lambd
    lambd = 1.5
    net = Network(
        node_counts, activation_function_hidden,
        activation_derivative_function_hidden, lambd=lambd)
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
    # exit()

    heading('Network Training')
    epochs = 200
    alpha = 0.1
    beta = 0.9
    costs = net.run_gradient_descent(
        X=x_train, Y=y_train, epochs=epochs, alpha=alpha, beta=beta)

    heading('Costs')
    print(costs)

    heading('Testing: Training set')
    Y_predict = net.forward_propagate(x_train)
    m_test = y_train.shape[1]
    Y_thresh = Y_predict > 0.5
    y_test_transormed = (y_train == 1)
    score = np.sum(Y_thresh == y_test_transormed) / m_test
    print('Score: %s percent' % (score * 100))

    heading('Testing: Test set')
    Y_predict = net.predict(x_test)
    m_test = y_test.shape[1]
    Y_thresh = Y_predict > 0.5
    y_test_transormed = (y_test == 1)
    score = np.sum(Y_thresh == y_test_transormed) / m_test
    print('Score: %s percent' % (score * 100))

    mapping = {
        False: digits[0],
        True: digits[1]
    }

    indices_list = [i for i in range(25, 30)]
    show_samples(
        x=x_test, y=Y_thresh, indices_list=indices_list, mapping=mapping)
    parameters = 'alpha: %s, beta: %s' % (alpha, beta)
    plot_cost(costs, parameters)
    plt.show()

if __name__ == '__main__':
    test()
