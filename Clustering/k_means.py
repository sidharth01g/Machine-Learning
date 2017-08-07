import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from utils.common import heading


def run():
    clusters = 4
    (X, y) = make_blobs(
        n_samples=200,
        n_features=2,
        centers=clusters,
        cluster_std=1.5,
        shuffle=True,
        random_state=1
    )
    heading('Data')
    print('X:', X.shape)
    print('y:', y.shape)

    # Scatter plot
    for class_ in np.unique(y):
        plt.scatter(
            X[y == class_, 0],
            X[y == class_, 1]
        )

    plt.grid()

    k_means = KMeans(
        n_clusters=clusters,
        init='k-means++',
        n_init=5,
        max_iter=500,
        tol=1e-03,
        random_state=123
    )

    y_kmeans = k_means.fit_predict(X)

    pp.pprint(y_kmeans)
    plt.title('Original samples')
    plt.figure()
    # Scatter plot
    for class_ in np.unique(y_kmeans):
        plt.scatter(
            X[y_kmeans == class_, 0],
            X[y_kmeans == class_, 1]
        )

    plt.grid()
    plt.title('Clustered sample points')
    plt.show()


if __name__ == '__main__':
    run()
