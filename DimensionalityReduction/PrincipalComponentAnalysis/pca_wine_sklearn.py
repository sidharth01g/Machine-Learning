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

    # Perform Principal Component Analysis on training feature vectors
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Classify using logistic regression
    lr = LogisticRegression()
    lr.fit(X_train_pca, y_train)
    # y_test = lr.predict(X_test_pca)

    # Plot the training set
    custom_plot = plots.CustomPlots()
    custom_plot.plot_decision_regions_1(X_train_pca, y_train, classifier=lr)

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc='best')
    plt.title('Training set')
    plt.show()

    # Plot the test set
    custom_plot.plot_decision_regions_1(X_test_pca, y_test, classifier=lr)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(loc='best')
    plt.title('Test set')
    plt.show()


if __name__ == '__main__':
    test()
