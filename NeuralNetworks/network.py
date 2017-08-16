import numpy as np
import pprint as pp
import pyprind
import random


class NeuralNetwork(object):

    def __init__(self, node_counts):
        self.node_counts = node_counts
        self.weights = np.array(
            [
                np.random.randn(self.node_counts[i + 1], self.node_counts[i])
                for i in range(self.n_layers)[: -1]
            ]
        )
        self.biases = np.array(
            [
                np.random.randn(self.node_counts[i])
                for i in range(self.n_layers)[1:]
            ]
        )

    @property
    def n_layers(self):
        return len(self.node_counts)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))

    @staticmethod
    def cost_derivative(output_activations, y):
        return (output_activations - y)

    def feed_forward(self, x):
        for i in range(self.n_layers - 1):
            print('\nfeed_forward stage: ', i)
            print(
                'w: ', self.weights[i].shape,
                ',  x: ', x.shape,
                ',  b: ', self.biases[i].shape
            )
            x = NeuralNetwork.sigmoid(
                np.dot(self.weights[i], x)
                + self.biases[i][np.newaxis].T
            )
        return x

    def back_propagate(self, x, y):
        nabla_b = np.array(
            [
                np.zeros(b.shape) for b in self.biases
            ]
        )
        nabla_w = np.array(
            [
                np.zeros(w.shape) for w in self.weights
            ]
        )

        activation = x
        activations = [activation]
        zs = []

        # Feed forward and store 'z' and activations to be consumed by
        # back-propagation
        for i in range(self.n_layers - 1):
            z = (
                np.dot(self.weights[i], activation)[np.newaxis].T
                + self.biases[i][np.newaxis].T
            ).flatten()

            zs.append(z)
            activation = NeuralNetwork.sigmoid(z)
            activations.append(activation)

        activations = np.array(activations)

        # Back-propagation
        delta = (
            NeuralNetwork.cost_derivative(activations[-1], y)
            * NeuralNetwork.sigmoid_prime(zs[-1])
        )

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta[np.newaxis].T, activations[-2][np.newaxis])

        for l in range(2, self.n_layers):
            delta = (
                np.dot(self.weights[-l + 1].T, delta)
                * NeuralNetwork.sigmoid_prime(zs[-l])
            )
            nabla_b[-l] = delta
            # nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
            nabla_w[-l] = np.dot(
                delta[np.newaxis].T, activations[-l - 1][np.newaxis])

        return (nabla_b, nabla_w)

    def update_minibatch(self, mini_batch, eta):
        nabla_b = np.array(
            [
                np.zeros(b.shape) for b in self.biases
            ]
        )
        nabla_w = np.array(
            [
                np.zeros(w.shape) for w in self.weights
            ]
        )

        for x, y in zip(mini_batch[0], mini_batch[1]):
            (delta_nabla_b, delta_nabla_w) = self.back_propagate(x, y)
            nabla_b += delta_nabla_b
            nabla_w += delta_nabla_w

        for i in range(self.n_layers - 1):
            self.weights[i] -= (
                eta / len(mini_batch)
                * nabla_w[i]
            )
            self.biases[i] -= (
                eta / len(mini_batch)
                * nabla_b[i]
            )

    def stochastic_gradient_descent(self, x_train, y_train,
                                    batch_size, eta, epochs):

        training_data = [[x, y] for x, y in zip(x_train, y_train)]
        n_samples = len(training_data)

        progress_bar = pyprind.ProgBar(epochs * n_samples, monitor=True,
                                       title='Neural Network Training')

        for epoch in range(epochs):
            print('\n\nEpoch: ', epoch)
            random.shuffle(training_data)
            start_index = 0

            while(start_index < n_samples):
                stop_index = min(start_index + batch_size, n_samples)

                x_temp = np.array(
                    [
                        data[0]
                        for data in training_data[start_index: stop_index]
                    ]
                )
                y_temp = np.array(
                    [
                        data[1]
                        for data in training_data[start_index: stop_index]
                    ]
                )
                for _ in range(start_index, stop_index):
                    progress_bar.update()
                mini_batch = (x_temp, y_temp)
                self.update_minibatch(mini_batch, eta)
                start_index = stop_index


def test():
    # Perform import specific to test() method
    import os
    import sys
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    sys.path.insert(0, parent_dir)
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.common import heading
    from utils.common import load_mnist_dataset

    # Prepare training data
    heading('Data preparation')
    data_dir = os.path.join(parent_dir, 'downloads', 'MNIST')
    print('Dataset directory: ', data_dir)
    (X_train, y_train) = load_mnist_dataset(data_dir)
    heading('Trainiing data')
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)

    # Instantiate neural network
    node_counts = [28*28, 20, 10]
    net = NeuralNetwork(node_counts)
    pp.pprint(net)
    print('Layers: ', net.n_layers)
    print('node_counts: ', net.node_counts)
    print('Weights:')
    pp.pprint(net.weights)
    pp.pprint(net.weights[0].shape)
    print('Biases:')
    pp.pprint(net.biases)
    pp.pprint(net.biases[0].shape)

    # Test feed_forward
    heading('Feed-forward test')
    x_sample = X_train[1][np.newaxis].T
    y_sample = y_train[1]
    print(x_sample.shape)
    result = net.feed_forward(x_sample)
    pp.pprint(result)
    pp.pprint(result.shape)

    # heading('Test back propagation')
    # net.back_propagate(x_sample, y_sample)

    heading('Test update_minibatch')
    print(y_train[:5].shape)
    """
    print(zip(X_train[:5], y_train[:5]))
    for x, y in zip(X_train[:5], y_train[:5]):
        print(type(x), type(y))
        print(x.shape, y)
    """
    """
    net.update_minibatch(
        mini_batch=(X_train[:5], y_train[:5],),
        eta=0.1
    )
    for w in net.weights:
        pp.pprint(w)
    """
    heading('Test stochastic_gradient_descent')
    net.stochastic_gradient_descent(
        x_train=X_train,
        y_train=y_train,
        batch_size=10,
        eta=0.1,
        epochs=2
    )

    exit('TEST')


if __name__ == '__main__':
    test()
