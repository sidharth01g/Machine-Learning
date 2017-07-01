from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import utils
import utils.plots
import matplotlib


def test1():
    # Load dataset
    iris = datasets.load_iris()

    # Extract 2-dimensional feature vectors from dataset
    X = iris.data[:, [2, 3]]

    # Load class labels
    y = iris.target[:]

    # Split dataset into training and test data portions
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Scale feature vectors
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # misclassification_penalties = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    gamma_list = [float(2**i) for i in range(10)]

    # for c in misclassification_penalties:
    for g in gamma_list:

        # Run SVM
        svm = SVC(
            kernel="rbf",
            gamma=g,
            C=10.0,
            random_state=0)
        svm.fit(X_train_std, y_train)  # Train SVM

        # Combine scaled training and testing feature vectors
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        # Plot decision regions
        # matplotlib.pyplot.figure()
        custom_plot1 = utils.plots.CustomPlots()
        custom_plot1.plot_decision_regions_1(
            X=X_combined_std,
            y=y_combined,
            classifier=svm,
            test_idx=range(105, 150))

        matplotlib.pyplot.xlabel("x1 (standardized)")
        matplotlib.pyplot.ylabel("x2 (standardized)")
        matplotlib.pyplot.legend(loc="upper left")
        # matplotlib.pyplot.close()

    matplotlib.pyplot.show()


if __name__ == "__main__":
    test1()
