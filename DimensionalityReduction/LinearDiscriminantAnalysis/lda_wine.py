from datasets_custom import WineExample
import matplotlib.pyplot as plt
import numpy as np
import plots
import pprint as pp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
    X_test_std = sc.transform(X_test)

    class_labels = np.unique(wine.df.iloc[:, 0])
    class_means = []

    heading('Class means: ')

    for label in class_labels:
        mean_ = np.mean(X_train_std[y_train == label], axis=0)
        print('Class label ', label, ':\n\t', mean_, '\n')
        class_means.append(mean_)

    # Initialize within-class scatter matrix to zeros
    d = X_train_std.shape[1]
    s_w = np.zeros((d, d))

    """
    # The following loops do not scale the individual within-class scatter
    # matrices by the number of features in that class
    for label, class_mean in zip(class_labels, class_means):

        for x in X_train_std[y_train == label]:
            X_train_std_zero_mean = x - class_mean[label]
            X_train_std_zero_mean_t = X_train_std_zero_mean.reshape(d, 1)
            s_w += X_train_std_zero_mean_t.dot(X_train_std_zero_mean_t.T)
    """

    # The total within-class scatter matrix = sum of covariance matrices
    # if scaled
    for label, class_mean in zip(class_labels, class_means):

        class_scatter_matrix = np.cov(X_train_std[y_train == label].T)
        s_w += class_scatter_matrix
        heading("Class %s within-class scatter matrix: " % str(label))
        pp.pprint(class_scatter_matrix)
        print('Shape', class_scatter_matrix.shape)

    heading('Within-class scatter matrix (Sw): ')
    pp.pprint(s_w)
    print('\nShape: ', s_w.shape)

    # Initialize between-class scatter matrix to zeros
    s_b = np.zeros((d, d))

    # Mean vector of all the training samples
    global_mean = np.sum(X_train_std, axis=0)
    global_mean = global_mean.reshape((d, 1))
    heading('Mean of all training features: (zero if standard-scaled)')
    pp.pprint(global_mean)
    print('Shape: ', global_mean.shape)

    # Compute the between-class scatter matrix
    for label, class_mean in zip(class_labels, class_means):
        class_mean = class_mean.reshape((d, 1))
        mean_diff = class_mean - global_mean
        class_count = X_train_std[y_train == label].shape[0]
        s_b += class_count * mean_diff.dot(mean_diff.T)

    heading('Between-class scatter matrix:')
    pp.pprint(s_b)
    print('Shape: ', s_b.shape)


if __name__ == '__main__':
    test()
