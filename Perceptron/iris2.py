from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)
import utils
import utils.plots
import matplotlib


def test():
    # Load Iris dataset
    iris = datasets.load_iris()

    # Load an array of 2D feature vectors
    X = iris.data[:, [2, 3]]
    # print(X)
    # print(type(X))

    # Load class labels
    y = iris.target[:]
    # print(y)

    # Split up training and test samples
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Scale the feature vectors
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    print(X_train_std)
    print(X_test_std)

    perceptron = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    perceptron.fit(X_train_std, y_train)

    y_predicted = perceptron.predict(X_test_std)

    test_samples_count = len(y_test)
    misclassifications_count = (y_predicted != y_test).sum()
    accuracy = 100.0 * (test_samples_count - misclassifications_count) / (
        test_samples_count
    )

    print("Number of test samples: ", test_samples_count)
    print("Number of samples misclassified: ", misclassifications_count)
    print("Accuracy: %s percent" % str(accuracy))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    custom_plot1 = utils.plots.CustomPlots()
    custom_plot1.plot_decision_regions_1(
        X=X_combined_std,
        y=y_combined,
        classifier=perceptron,
        test_idx=range(105, 150))
    matplotlib.pyplot.xlabel("Petal length (standardized)")
    matplotlib.pyplot.ylabel("Petal width (standardized)")
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()









if __name__ == "__main__":
    test()
