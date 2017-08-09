import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import pprint as pp
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
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

    heading('Agglomerative Clustering')
    agg_clustering = AgglomerativeClustering(
        n_clusters=2,
        affinity='euclidean',
        memory=dir_path,
        linkage='complete',
    )

    y_agg = agg_clustering.fit_predict(X)
    print('\nClusters:')
    pp.pprint(y_agg)
    pp.pprint(y_agg.shape)

    ax2.scatter(
        X[y_agg == 0, 0],
        X[y_agg == 0, 1],
        label='Cluster 0'
    )
    ax2.scatter(
        X[y_agg == 1, 0],
        X[y_agg == 1, 1],
        label='Cluster 1'
    )
    ax2.set_title('Agglomerative Clustering')

    heading('DBSCAN Clustering')

    dbscan = DBSCAN(
        eps=0.3,
        min_samples=5,
        metric='euclidean',
        algorithm='auto',
        n_jobs=-1
    )

    plt.figure()

    y_dbscan = dbscan.fit_predict(X)
    print('\nClusters:')
    pp.pprint(y_dbscan)
    pp.pprint(y_dbscan.shape)
    plt.scatter(
        X[y_dbscan == 0, 0],
        X[y_dbscan == 0, 1],
        label='Cluster 0'
    )
    plt.scatter(
        X[y_dbscan == 1, 0],
        X[y_dbscan == 1, 1],
        label='Cluster 1'
    )
    plt.title('DBSCAN Clustering')
    plt.show()


if __name__ == '__main__':
    run()
