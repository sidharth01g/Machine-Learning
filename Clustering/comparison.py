import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint as pp
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.clustering import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from utils.common import heading


def run():
    (X, y) = make_moons(n_samples=200, noise=0.05, random_state=1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Original data')
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    # plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2)

    n_clusters = 2
    heading('Data')
    print(X.shape)

    heading('K-Means Clustering')
    k_means = KMeans(
        n_clusters=n_clusters,
        max_iter=500,
        init='k-means++',
        algorithm='auto',
        precompute_distances=True,
        tol=1e-03,
        n_jobs=-1,
        random_state=1,
        verbose=1,
        copy_x=True
    )

    y_kmeans = k_means.fit_predict(X)
    print('\nClusters:')
    pp.pprint(y_kmeans)
    pp.pprint(y_kmeans.shape)

    ax1.scatter(
        X[y_kmeans == 0, 0],
        X[y_kmeans == 0, 1],
        label='Cluster 0'
    )
    ax1.scatter(
        X[y_kmeans == 1, 0],
        X[y_kmeans == 1, 1],
        label='Cluster 1'
    )
    ax1.set_title('K Means Clustering')
    plt.show()



if __name__ == '__main__':
    run()
