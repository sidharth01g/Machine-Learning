from datasets_custom import WineExample
import numpy as np
import pprint as pp
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def heading(text, character='='):
    if type(text) is not str:
        heading('<INVALID_HEADING>')
        return
    print('\n')
    print(text)
    print(character * len(text))


def test():
    np.set_printoptions(suppress=True)
    wine = WineExample()
    wine.fetch_data()

    X = wine.df.iloc[:, 1:].values
    y = wine.df.iloc[:, 0].values

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Scale the feature vectors
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    heading('Standardized training feature vectors:')
    pp.pprint(X_train_std)
    print('Shape: ', X_train_std.shape)

    squared_distances = pdist(X_train_std, 'sqeuclidean')
    heading('Pairwise squared Eclidean distances:')
    pp.pprint(squared_distances)
    print('Shape: ', squared_distances.shape)

    squared_distances_matrix = squareform(squared_distances)
    heading('Squared distances matrix:')
    pp.pprint(squared_distances_matrix)
    print('Shape: ', squared_distances_matrix.shape)

    # Compute symmetric kernel matrix
    gamma = 1
    K = exp(-gamma * squared_distances_matrix)
    heading('Kernel matrix: ')
    pp.pprint(K)
    print('Shape: ', K.shape)

    # Center the kernel matrix using formula
    n_vectors = K.shape[0]
    ones_n = (np.ones((n_vectors, n_vectors))) / n_vectors
    K = K - ones_n.dot(K) - K.dot(ones_n) + ones_n.dot(K).dot(ones_n)
    heading('Kernel matrix (Centered): ')
    pp.pprint(K)
    print('Shape: ', K.shape)

    # compute eigenvalues and eigenvectors
    n_components = 3
    heading('Largest eigenvalues: (n=%s)' % str(n_components))
    (eigenvalues, eigenvectors) = eigh(K)

    # Print the largest "n_components" number of eigenvalues
    for i in range(1, n_components + 1):
        print('Eigenvalue (%s):' % str(i), eigenvalues[-i])
        print('Eigenvector component (%s):' % str(i), eigenvectors[-i])
        print('Shape: ', eigenvectors[i].shape, '\n')

    X_principal_components = np.column_stack(
        (eigenvectors[-i] for i in range(1, n_components + 1))
    )

    heading('Final Kernel PCA vectors:')
    pp.pprint(X_principal_components)
    print('Shape: ', X_principal_components.shape)


if __name__ == '__main__':
    test()
