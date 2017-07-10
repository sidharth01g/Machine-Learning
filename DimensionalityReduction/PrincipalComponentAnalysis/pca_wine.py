from datasets_custom import WineExample
from matplotlib import pyplot as plt
import numpy as np
import pprint as pp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    print("\nShape of standardized input vectors array:")
    pp.pprint(X_train_std.shape)

    # Compute the covariance matrix for X_train_std
    covariance_matrix = np.cov(X_train_std.T)
    print("\nCovariance matrix of standardized feature vectors:")
    pp.pprint(covariance_matrix)

    # Compute Eigen values and Eigen vectors
    (eigen_values, eigen_vectors) = np.linalg.eig(covariance_matrix)

    print('\nEigen values of the covariance matrix:')
    pp.pprint(eigen_values)

    print('\nEigen vectors of the covariance matrix:')
    pp.pprint(eigen_vectors)

    # Compute the "variance explained ratios"
    eigen_values_sum = sum(eigen_values)
    variance_explained_ratios = [
        (eigen_value / eigen_values_sum) for eigen_value in sorted(
            eigen_values, reverse=True  # descending order
        )
    ]
    variance_explained_ratios_cumulative = np.cumsum(variance_explained_ratios)

    print("\nSum of Eigen values: ", eigen_values_sum)
    print('\nVariance explained ratios:')
    pp.pprint(variance_explained_ratios)
    print("\nCumulative variance explained ratios:")
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

    # Create the projection matrix by stacking the pricipal Eigen vectors
    # horizontally
    w = np.hstack((w_pc1, w_pc2))
    print('\nProjection matrix:')
    pp.pprint(w)


if __name__ == '__main__':
    test()
