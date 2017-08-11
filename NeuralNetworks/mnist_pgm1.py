import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import pprint as pp
from utils.common import heading
from utils.common import load_mnist_dataset


def run():
    data_dir = os.path.join(parent_dir, 'downloads', 'MNIST')
    print('Dataset directory: ', data_dir)
    (X_train, y_train) = load_mnist_dataset(data_dir)
    heading('Trainiing data')
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)

    (fig, ax) = plt.subplots(nrows=2, ncols=5, sharex=True)
    ax = ax.flatten()
    for digit in range(10):
        # plt.figure()
        digit_serial = X_train[y_train == digit][0]
        digit_reshaped = digit_serial.reshape(28, 28)
        pp.pprint(digit_reshaped)
        ax[digit].imshow(digit_reshaped, cmap='Greys', interpolation='nearest')
        ax[digit].set_xticks([])
        ax[digit].set_yticks([])

    plt.show()


if __name__ == '__main__':
    run()
