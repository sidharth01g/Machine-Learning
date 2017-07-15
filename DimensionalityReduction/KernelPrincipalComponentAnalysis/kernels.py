import numpy as np
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def rbf_kernel_pca(X, gamma, n_components):

    # Compute squared distance pairs
    squared_distances = pdist(X, 'sqeuclidean')
    squared_distances_matrix = squareform(squared_distances)

    # Compute symmetric kernel matrix
    K = exp(-gamma * squared_distances_matrix)

    # Center the kernel matrix using formula
    n_vectors = K.shape[0]
    ones_n = np.ones((n_vectors, n_vectors)) / n_vectors
    K = K - ones_n.dot(K) - K.dot(ones_n) + ones_n.dot(K).dot(ones_n)
    (eigenvalues, eigenvectors) = eigh(K)

    # Pick the largest eigenvalues (last value is the largest)
    X_principal_components = np.column_stack(
        (eigenvectors[:, -i] for i in range(1, n_components + 1))
    )

    return X_principal_components
