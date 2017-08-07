import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples

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

    heading('Prediction:')
    y_kmeans = k_means.fit_predict(X)
    pp.pprint(y_kmeans)
    pp.pprint(y_kmeans.shape)

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
    # plt.show()

    cluster_labels = np.unique(y_kmeans)
    heading('Cluster labels')
    print(cluster_labels)
    n_clusters = cluster_labels.shape[0]

    heading('Silhouette values for each sample:')
    silhouette_values = silhouette_samples(
        X,
        y_kmeans,
        metric='euclidean'
    )
    pp.pprint(silhouette_values)
    pp.pprint(silhouette_values.shape)

    y_bottom = 0
    y_ticks = []
    heading('Silhouette plot')
    plt.figure()
    for index, label in enumerate(cluster_labels):
        silhouette_values_cluster = silhouette_values[y_kmeans == label]
        silhouette_values_cluster.sort()
        print('\nSilhouette values for label ', label)
        pp.pprint(silhouette_values_cluster)
        pp.pprint(silhouette_values_cluster.shape)
        color = cm.jet(index / n_clusters)

        y_top = y_bottom + len(silhouette_values_cluster)
        plt.barh(
            range(y_bottom, y_top),
            silhouette_values_cluster,
            color=color,
            edgecolor='none',
            height=1.0
        )
        y_ticks.append((y_bottom + y_top) / 2)
        y_bottom = y_top

    plt.title('Silhouette plot')
    plt.xlabel('Silhouette value')
    plt.ylabel('Cluster')
    plt.yticks(y_ticks, cluster_labels)
    plt.show()


if __name__ == '__main__':
    run()
