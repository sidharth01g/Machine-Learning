from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

    # print(X_train_std)
    # print(X_test_std)

    log_reg = LogisticRegression(C=1000.0, random_state=0)
    log_reg.fit(X_train_std, y_train)

    y_predicted = log_reg.predict(X_test_std)

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
        classifier=log_reg,
        test_idx=range(105, 150))
    matplotlib.pyplot.xlabel("Petal length (standardized)")
    matplotlib.pyplot.ylabel("Petal width (standardized)")
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    print(X_test_std[0])
    print(X_test_std[0, :])
    probabilities = log_reg.predict_proba(X_test_std[0, :].reshape(1,-1))
    print("Probabilites: ", probabilities * 100)


def plot_variation():
    """Plots the variation of weights for
    """
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

    # Compute weights for varying values of Inverse Regularization Parameter C
    weights_list = []
    c_list = []

    for i in range(-5, 6):
        log_reg = LogisticRegression(C=10**i, random_state=0)
        log_reg.fit(X_train_std, y_train)
        # Store weights for class 2 (index =1) in a list
        weights_list.append(log_reg.coef_[1])
        c_list.append(10**i)
        print(10**i)
        print(log_reg.coef_[1])
    print("*" * 100)
    weights_array = np.array(weights_list)
    print(weights_array)
    print(weights_array[:, 0])
    print(weights_array[:, 1])

    matplotlib.pyplot.plot(
        c_list, weights_array[:, 0], label='Weight for Petal Length')
    matplotlib.pyplot.plot(
        c_list, weights_array[:, 1], label='Weight for Petal Width')
    matplotlib.pyplot.ylabel("Weight coefficients")
    matplotlib.pyplot.xlabel("Inverse regularization parameter (C)")
    matplotlib.pyplot.legend("upper left")
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.show()


if __name__ == "__main__":
    test()
    plot_variation()
