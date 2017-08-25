from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
import pyprind

np.set_printoptions(threshold=np.nan)


class Neuron(object):

    def __init__(self, dimensions):
        assert(type(dimensions) is int)
        self.w = np.zeros((dimensions, 1))
        self.w = np.random.rand(dimensions, 1)
        self.b = 0.0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def propagate(self, X, Y):
        m = X.shape[0]
        Z = np.dot(self.w.T, X) + self.b
        A = Neuron.sigmoid(Z)

        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)

        gradients = {
            'dw': dw,
            'db': db
        }
        # print(A)
        cost = (-1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return (gradients, cost)

    def run_gradient_descent(self, X, Y, eta, n_iter):

        costs = []
        block_character = bytes((219,)).decode('cp437')
        progress_bar = pyprind.ProgBar(
            n_iter, monitor=True, title='Neuron Training',
            bar_char=block_character
        )
        for i in range(n_iter):
            (gradients, cost) = self.propagate(X, Y)
            costs.append(cost)

            # update parameters
            self.w -= eta * gradients['dw']
            self.b -= eta*gradients['db']

            progress_bar.update()
        return costs

    def predict(self, X):
        Z = np.dot(self.w.T, X) + self.b
        A = Neuron.sigmoid(Z)

        m = X.shape[1]
        Y_predict = np.zeros((1, m))
        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                Y_predict[0, i] = 0
            else:
                Y_predict[0, i] = 1
        return Y_predict


def test():
    # Perform import specific to test() method
    import os
    import sys
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    sys.path.insert(0, parent_dir)
    import numpy as np
    from utils.common import heading
    from utils.common import load_mnist_dataset

    # Prepare training data
    heading('Data preparation')
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
    print(X.shape)
    print(y.shape)

    X = X / np.amax(X)

    train_ratio = 0.7
    train_size = int(train_ratio * X.shape[1])

    x_train = X[:, :train_size]
    y_train = y[:, :train_size]

    x_test = X[:, train_size:]
    y_test = y[:, train_size:]
    """
    for index in [10, 30, 60, 70, 75]:
        digit_serial = x_test[:, index]
        digit_reshaped = digit_serial.reshape(28, 28)
        # pp.pprint(digit_reshaped)
        plt.figure()
        plt.imshow(digit_reshaped, cmap='Greys', interpolation='nearest')
        plt.title(str(y_test[0, index]))
    plt.show()
    """
    print('Training samples: ', train_size)
    print('Training vectors: ', x_train.shape)
    print('Training classes', y_train.shape)
    print('Test vectors: ', x_test.shape)
    print('Test classes', y_test.shape)

    heading('Neuron')
    dim = X.shape[0]
    neuron = Neuron(dim)
    print('Weights: ', neuron.w.shape[0])
    (gradients, cost) = neuron.propagate(x_train, y_train)
    print('Weight gradients:')
    pp.pprint(gradients['dw'].shape)
    print('Bias gradients:')
    pp.pprint(gradients['db'].shape)

    heading('Training neuron: Gradent descent')
    costs = neuron.run_gradient_descent(
        X=x_train, Y=y_train, eta=0.1, n_iter=100)
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.show()

    heading('Testing neuron')
    Y_predict = neuron.predict(x_test)
    accuracy = 1.0 - (np.sum(np.abs(Y_predict - y_test)) / y_test.shape[1])
    print('Accuracy: %s percent' % str(accuracy * 100))


if __name__ == '__main__':
    test()
