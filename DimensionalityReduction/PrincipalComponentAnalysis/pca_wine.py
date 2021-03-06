from datasets_custom import WineExample
from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
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

    heading("\nShape of standardized input vectors array:")
    pp.pprint(X_train_std.shape)

    # Compute the covariance matrix for X_train_std
    covariance_matrix = np.cov(X_train_std.T)
    heading("\nCovariance matrix of standardized feature vectors:")
    pp.pprint(covariance_matrix)

    # Compute Eigen values and Eigen vectors
    (eigen_values, eigen_vectors) = np.linalg.eig(covariance_matrix)

    heading('\nEigen values of the covariance matrix:')
    pp.pprint(eigen_values)

    heading('\nEigen vectors of the covariance matrix:')
    pp.pprint(eigen_vectors)

    # Compute the "variance explained ratios"
    eigen_values_sum = sum(eigen_values)
    variance_explained_ratios = [
        (eigen_value / eigen_values_sum) for eigen_value in sorted(
            eigen_values, reverse=True  # descending order
        )
    ]
    variance_explained_ratios_cumulative = np.cumsum(variance_explained_ratios)

    heading("\nSum of Eigen values: ", eigen_values_sum)
    heading('\nVariance explained ratios:')
    pp.pprint(variance_explained_ratios)
    heading("\nCumulative variance explained ratios:")
    pp.pprint(variance_explained_ratios_cumulative)

    plt.bar(
        range(1, len(variance_explained_ratios) + 1),
        variance_explained_ratios,
        alpha=0.5,
        align='center',
        label='Explained variances'
    )

    plt.step(
        range(1, len(variance_explained_ratios) + 1),
        variance_explained_ratios_cumulative,
        where='mid',
        label='Cumulative explained variance'
    )

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Pricipal components index')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # Contruct Eigen pairs as a list of (eigen_value, eigen_vector) tuples
    eigen_pairs = [
        (np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(
            len(eigen_values)
        )
    ]

    # Sort Eigen pairs in the asceding order of Eigen values
    eigen_pairs.sort(reverse=True)

    # Consider two principal components (2 highest Eigen values) out of 13
    w_pc1 = eigen_pairs[0][1][:, np.newaxis]
    w_pc2 = eigen_pairs[2][1][:, np.newaxis]

    # Create the projection matrix 'w' by stacking the pricipal Eigen vectors
    # horizontally
    w = np.hstack((w_pc1, w_pc2))
    heading('\nProjection matrix:')
    pp.pprint(w)

    heading('Matrix shapes:')
    print('X_train_std: ', X_train_std.shape)
    print('Projection matrix: ', w.shape)

    # Compute PCA-reduced feature vectors by computing dot product of the
    # d-dimensional vectors with the projection matrix
    X_train_reduced = X_train_std.dot(w)
    heading('Dimensionality-reduced feature vectors: ')
    pp.pprint(X_train_reduced)

    # Plot the reduced 2d feature vectors
    class_settings = [
        (1, 'r', 's'),
        (2, 'b', 'x'),
        (3, 'g', 'o')
    ]

    for (class_index, color, marker) in class_settings:
        plt.scatter(
            X_train_reduced[y_train==class_index, 0],
            X_train_reduced[y_train==class_index, 1],
            c=color,
            label=class_index,
            marker=marker
        )
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    test()
