import numpy as np
import pprint as pp


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
                np.dot(self.weights[i], activation)
                + self.biases[i][np.newaxis].T
            )
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
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.n_layers):
            delta = (
                np.dot(self.weights[-l + 1].T, delta)
                * NeuralNetwork.sigmoid_prime(zs[-l])
            )
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
            # pp.pprint(nabla_b[-l])
            # pp.pprint(nabla_w[-l])

        return (nabla_b, nabla_w)


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
    x_sample = X_train[0][np.newaxis].T
    y_sample = y_train[0]
    print(x_sample.shape)
    result = net.feed_forward(x_sample)
    pp.pprint(result)
    pp.pprint(result.shape)

    heading('Test back propagation')
    net.back_propagate(x_sample, y_sample)
    exit('TEST')


if __name__ == '__main__':
    test()
