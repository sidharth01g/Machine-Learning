import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import pprint as pp
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from utils import common_methods


def test():

    (X, y) = make_moons(n_samples=100, random_state=123)

    common_methods.heading('Feature vectors: ')
    pp.pprint(X)
    common_methods.heading('Labels: ')
    pp.pprint(y)

    kernel_pca = KernelPCA(
        n_components=2,
        kernel='rbf',
        gamma=15
    )

    X_pca = kernel_pca.fit_transform(X)
    plt.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        color='red',
        marker='^',
        alpha=0.5
    )
    plt.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        color='blue',
        marker='o',
        alpha=0.5
    )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


if __name__ == '__main__':
    test()
