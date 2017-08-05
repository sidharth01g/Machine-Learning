import numpy as np
import pandas
import pprint as pp
import matplotlib.pyplot
import numpy
import sys
import os
from sklearn.cross_validation import train_test_split
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)
import utils
import utils.plots

class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        result = np.dot(X, self.w[1:]) + self.w[0]
        return result

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


    def fit(self, X, y):
        self.w = np.zeros(X.shape[1] + 1)
        self.cost = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = 0.5 * ((errors**2).sum())
            self.cost.append(cost)
        return self



def get_dataframe(url, header=None):

    try:
        dataframe_obj = pandas.read_csv(url, header)
    except Exception as error:
        print("Exception occured while getting dataframe")
        print(str(error))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Line: " + str(exc_tb.tb_lineno))
        raise error

    return dataframe_obj


def test_iris():

    url = (
        "https://"
        + "archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

    print("Fetching Iris dataset from: " + url)

    try:
        dataframe_obj = get_dataframe(url)
    except Exception as error:
        print("ERROR: Fetching dataframe failed")
        print(str(error))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Line: " + str(exc_tb.tb_lineno))
        raise error

    dataframe_obj = pandas.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/' +
        'iris.data', header=None)
    # print("Printing Iris dataset features:")
    # print(dataframe_obj)

    print("\nPrinting classes:")
    y = dataframe_obj.iloc[0:100, 4].values

    print("\nPrinting y:")
    y = numpy.where(y == "Iris-setosa", 1, -1)

    print("Printing x:")
    X = dataframe_obj.iloc[0:100, [0, 2]].values
    print(X)
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    ad = AdalineGD()
    ad.fit(X_train, y_train)
    pp.pprint(ad.cost)



if __name__ == "__main__":
    test_iris()
