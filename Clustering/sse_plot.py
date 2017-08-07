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
    clusters = 5
    (X, y) = make_blobs(
        n_samples=300,
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

    plt.figure()

    total_within_class_sse_list = []
    cluster_centers_list = range(1, 11)

    for i in cluster_centers_list:

        k_means = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=5,
            max_iter=500,
            tol=1e-03,
            random_state=123
        )

        k_means.fit(X)
        total_within_class_sse = k_means.inertia_
        total_within_class_sse_list.append(total_within_class_sse)

    plt.plot(
        cluster_centers_list,
        total_within_class_sse_list,
        marker='.'
    )
    plt.xlabel('Number of cluster centers')
    plt.ylabel('Distortion (total within-class-SSE)')
    plt.title('Variation of distortion with within-class-SSE')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    run()
