import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
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
        # pp.pprint(digit_reshaped)
        ax[digit].imshow(digit_reshaped, cmap='Greys', interpolation='nearest')
        ax[digit].set_xticks([])
        ax[digit].set_yticks([])

    # Select a few samples (n_samples) of 'digit' for displaying
    digit = 5
    n_samples = 24
    digit_images = X_train[y_train == digit]
    digit_samples = digit_images[:n_samples]

    # Set number of rows and columns for subplots
    n_rows = int(np.floor(np.sqrt(n_samples)))
    n_columns = int(np.ceil(n_samples / n_rows))

    # Display the selected samples of 'digit'
    (fig, ax) = plt.subplots(nrows=n_rows, ncols=n_columns, sharex=True)
    ax = ax.flatten()
    index = 0

    for digit_serial in digit_samples:
        digit_reshaped = digit_serial.reshape(28, 28)
        ax[index].imshow(digit_reshaped, cmap='Greys', interpolation='nearest')

        ax[index].set_xticks([])
        ax[index].set_yticks([])
        index += 1

    heading('Digit: %s, Samples: %s' % (digit, n_samples))
    plt.show()


if __name__ == '__main__':
    run()
