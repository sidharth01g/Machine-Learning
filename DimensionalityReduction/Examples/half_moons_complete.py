import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from KernelPrincipalComponentAnalysis.kernels import rbf_kernel_pca2
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pprint as pp
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from utils import common_methods


def test():

    (X, y) = make_moons(n_samples=100, random_state=123)

    common_methods.heading('Feature vectors: ')
    pp.pprint(X)
    common_methods.heading('Labels: ')
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

    gamma = 15
    n_components = 2
    (alphas, lambdas) = rbf_kernel_pca2(
        X,
        gamma=gamma,
        n_components=n_components
    )

    common_methods.heading('Alphas: ')
    pp.pprint(alphas)
    print('Shape: ', alphas.shape)

    common_methods.heading('Lambdas: ')
    pp.pprint(lambdas)
    print('Length: ', len(lambdas))

    # Project a test sample
    test_sample_index = 30
    test_sample = X[test_sample_index]
    test_sample_projected_projected = alphas[test_sample_index]

    test_sample_reprojected = common_methods.project_x(
        test_sample,
        X,
        gamma=gamma,
        alphas=alphas,
        lambdas=lambdas
    )

    plt.scatter(
        alphas[y == 0, 0],
        np.zeros((50, 1)),
        color='red',
        marker='^',
        alpha=0.5
    )

    plt.scatter(
        alphas[y == 1, 0],
        np.zeros((50, 1)),
        color='blue',
        marker='o',
        alpha=0.5
    )


    plt.show()


if __name__ == '__main__':
    test()
