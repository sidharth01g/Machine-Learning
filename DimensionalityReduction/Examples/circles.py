import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from KernelPrincipalComponentAnalysis.kernels import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA


def test():
    n_samples = 1000
    (X, y) = make_circles(
        n_samples=n_samples,
        random_state=200,
        noise=0.1,
        factor=0.2
    )
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
    plt.title('Feature vectors from circle dataset (with noise)')

    pca = PCA(n_components=2)
    X_pca_sklearn = pca.fit_transform(X)

    (fig, ax) = plt.subplots(
        nrows=1,
        ncols=2,
    )

    fig.suptitle('Plain PCA')

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
        np.zeros((int(n_samples / 2), 1)),
        color='red',
        marker='^',
        alpha=0.5
    )

    ax[1].scatter(
        X_pca_sklearn[y == 1, 0],
        np.zeros((int(n_samples / 2), 1)),
        color='blue',
        marker='o',
        alpha=0.5
    )

    n_components = 2
    for gamma in np.linspace(0, 2, 10):

        X_kernel_pca = rbf_kernel_pca(
            X,
            gamma=gamma,
            n_components=n_components
        )
        (fig, ax) = plt.subplots(
            nrows=1,
            ncols=2,
        )
        fig.suptitle('RBF Kernel PCA - Gamma: %s' % str(gamma))
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
            np.zeros((int(n_samples / 2), 1)),
            color='red',
            marker='^',
            alpha=0.5
        )

        ax[1].scatter(
            X_kernel_pca[y == 1, 0],
            np.zeros((int(n_samples / 2), 1)),
            color='blue',
            marker='o',
            alpha=0.5
        )

    plt.show()


if __name__ == '__main__':
    test()
