from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import utils
import utils.plots


def test():
    np.random.seed(0)
    # Create 200 random valued inputs of size 2 (one for each input to XOR)
    x = np.random.randn(200, 2)

    # True for positive samples, False for negative samples
    a = (x[:, 0] > 0)
    b = (x[:, 1] > 0)

    # Output: XOR of input samples
    out_xor = np.logical_xor(a, b)

    # Convert True to +1, False to -1
    y_xor = np.where(out_xor, 1, -1)

    # Postive class: those input vectors for which y_xor is +1
    # Negative class: those input vectors for which y_xor is -1
    postive_vectors = x[y_xor == 1]
    negative_vectors = x[y_xor == -1]

    # Scatter plot of both classes
    plt.scatter(
        postive_vectors[:, 0], postive_vectors[:, 1],
        c='b', marker='+', label='+1')
    plt.scatter(
        negative_vectors[:, 0], negative_vectors[:, 1],
        c='r', marker='_', label='-1')
    plt.show()

    # Classify using RBF kernel
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
    svm.fit(x, y_xor)

    # Plot decision regions
    custom_plot1 = utils.plots.CustomPlots()
    custom_plot1.plot_decision_regions_1(
        X=x,
        y=y_xor,
        classifier=svm)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper left")
    plt.show()
    plt.close()


def test1():
    np.random.seed(0)
    # Create 200 random valued inputs of size 2 (one for each input to XOR)
    x = np.random.randn(200, 2)

    # True for positive samples, False for negative samples
    a = (x[:, 0] > 0)
    b = (x[:, 1] > 0)

    # Output: XOR of input samples
    out_xor = np.logical_xor(a, b)

    # Convert True to +1, False to -1
    y_xor = np.where(out_xor, 1, -1)

    # Postive class: those input vectors for which y_xor is +1
    # Negative class: those input vectors for which y_xor is -1
    postive_vectors = x[y_xor == 1]
    negative_vectors = x[y_xor == -1]

    # Scatter plot of both classes
    plt.scatter(
        postive_vectors[:, 0], postive_vectors[:, 1],
        c='b', marker='+', label='+1')
    plt.scatter(
        negative_vectors[:, 0], negative_vectors[:, 1],
        c='r', marker='_', label='-1')

    gamma_list = [float(2**i) for i in range(10)]
    for g in gamma_list:
        # plt.figure()
        # Classify using RBF kernel
        svm = SVC(kernel='rbf', random_state=0, gamma=g, C=5.0)
        svm.fit(x, y_xor)

        # Plot decision regions
        custom_plot1 = utils.plots.CustomPlots()
        custom_plot1.plot_decision_regions_1(
            X=x,
            y=y_xor,
            classifier=svm)

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(loc="upper left")

    plt.show()
    plt.close()


if __name__ == '__main__':
    test1()
