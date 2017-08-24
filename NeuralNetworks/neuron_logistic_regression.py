import numpy as np


class Neuron(object):

    def __init__(self, dimensions):
        if not dimensions:
            self.w = None
            self.b = None
        else:
            self.w = np.zeros((dimensions, 1))
            self.b = 0.0


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
    (X_train, y_train) = load_mnist_dataset(data_dir)
    print(X_train.shape)


if __name__ == '__main__':
    test()
