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
from utils.common import heading


def run():
    np.set_printoptions(threshold=np.nan)
    # Create a random dataframe
    np.random.seed(1)
    columns = ['X', 'Y', 'Z']
    indices = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4', ]

    X = np.random.random_sample(
        (len(indices), len(columns))
    )
    upper = 200
    lower = 100
    # Change the range of X to the interval [100 to 20)
    X = lower + (upper - lower) * X

    pp.pprint(X)
    pp.pprint(X.shape)

    heading('Dataframe:')

    df = pd.DataFrame(
        X,
        columns=columns,
        index=indices
    )

    pp.pprint(df)

    dist_condensed = pdist(df)

    heading('Condensed distance matrix')
    pp.pprint(dist_condensed)

    heading('Squareform distance matrix')
    dist_squareform = squareform(dist_condensed)
    pp.pprint(dist_squareform)

    heading('Distance dataframe')
    df_dist = pd.DataFrame(
        dist_squareform,
        columns=indices,
        index=indices
    )
    pp.pprint(df_dist)
    pp.pprint(df_dist.shape)

    heading('Clusters')
    # Complete linkage agglomeration
    method = 'complete'

    # Single linkage agglomeration
    # method = 'single'
    clusters = linkage(
        dist_condensed,
        method=method
    )
    pp.pprint(clusters)
    pp.pprint(clusters.shape)

    heading('Clustering dataframe')
    columns = [
        'Label 1',
        'Label 2',
        'Distance',
        'Number of points in cluster'
    ]
    cluster_indices = [
        'Cluster %d' % index for index in range(clusters.shape[0])
    ]

    df_clustering = pd.DataFrame(
        clusters,
        columns=columns,
        index=cluster_indices
    )

    pp.pprint(df_clustering)
    pp.pprint(df_clustering.shape)

    heading('Dendrogram')
    dendr = dendrogram(
        clusters,
        labels=indices,
    )
    pp.pprint(dendr)
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()


if __name__ == '__main__':
    run()
