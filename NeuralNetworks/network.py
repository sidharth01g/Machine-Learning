import numpy as np
import pprint as pp


class NeuralNetwork(object):

    def __init__(self, node_counts):
        self.node_counts = node_counts
        self.weights = np.array(
            [
                np.random.randn(self.node_counts[i], self.node_counts[i + 1])
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


def test():
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


if __name__ == '__main__':
    test()
