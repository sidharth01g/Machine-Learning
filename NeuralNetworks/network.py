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
    print(x_sample.shape)
    result = net.feed_forward(x_sample)
    pp.pprint(result)
    pp.pprint(result.shape)
    exit('TEST')


if __name__ == '__main__':
    test()
