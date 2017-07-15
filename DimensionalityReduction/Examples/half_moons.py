import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from KernelPrincipalComponentAnalysis.kernels import rbf_kernel_pca
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pprint as pp
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from utils.common_methods import heading


def test():

    (X, y) = make_moons(n_samples=100, random_state=123)

    heading('Feature vectors: ')
    pp.pprint(X)
    heading('Labels: ')
    pp.pprint(y)

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color='red',
        marker='^',
        alpha=0.5
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='blue',
        marker='o',
        alpha=0.5
    )

    pca_sklearn = PCA(n_components=2)
    X_pca_sklearn = pca_sklearn.fit_transform(X)

    (fig, ax) = plt.subplots(
        nrows=1,
        ncols=2,
    )

    ax[0].scatter(
        X_pca_sklearn[y == 0, 0],
        X_pca_sklearn[y == 0, 1],
        color='red',
        marker='^',
        alpha=0.5
    )

    ax[0].scatter(
        X_pca_sklearn[y == 1, 0],
        X_pca_sklearn[y == 1, 1],
        color='blue',
        marker='o',
        alpha=0.5
    )

    ax[1].scatter(
        X_pca_sklearn[y == 0, 0],
        np.zeros((50, 1)),
        color='red',
        marker='^',
        alpha=0.5
    )

    ax[1].scatter(
        X_pca_sklearn[y == 1, 0],
        np.zeros((50, 1)),
        color='blue',
        marker='o',
        alpha=0.5
    )

    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')

    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('Principal Component 1')

    # Kernel-PCA

    for gamma in range(1, 20, 2):
        X_kernel_pca = rbf_kernel_pca(X, gamma=gamma, n_components=2)

        heading('X_kernel_pca')
        pp.pprint(X_kernel_pca)

        (fig, ax) = plt.subplots(
            nrows=1,
            ncols=2,
        )

        ax[0].scatter(
            X_kernel_pca[y == 0, 0],
            X_kernel_pca[y == 0, 1],
            color='red',
            marker='^',
            alpha=0.5
        )

        ax[0].scatter(
            X_kernel_pca[y == 1, 0],
            X_kernel_pca[y == 1, 1],
            color='blue',
            marker='o',
            alpha=0.5
        )

        ax[1].scatter(
            X_kernel_pca[y == 0, 0],
            np.zeros((50, 1)),
            color='red',
            marker='^',
            alpha=0.5
        )

        ax[1].scatter(
            X_kernel_pca[y == 1, 0],
            np.zeros((50, 1)),
            color='blue',
            marker='o',
            alpha=0.5
        )

        ax[0].set_xlabel('Principal Component 1')
        ax[0].set_ylabel('Principal Component 2')

        ax[1].set_ylim([-1, 1])
        ax[1].set_yticks([])
        ax[1].set_xlabel('Principal Component 1')

        plt.title('Kernel PCA (gamma: %s)' % str(gamma))
        ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    plt.show()


if __name__ == '__main__':
    test()
