import pandas
import perceptron
import matplotlib.pyplot
import numpy
import sys


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
    print(y)

    print("\nPrinting y:")
    y = numpy.where(y == "Iris-setosa", 1, -1)
    print(y)

    print("Printing x:")
    x = dataframe_obj.iloc[0:100, [0, 2]].values
    print(x)
    print(type(x))

    matplotlib.pyplot.scatter(x[:50, 0], x[:50, 1], label="Setosa")
    matplotlib.pyplot.scatter(x[50:, 0], x[50:, 1], label="Versicolor")

    matplotlib.pyplot.xlabel("Petal length")
    matplotlib.pyplot.ylabel("Sepal length")

    matplotlib.pyplot.legend()

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    perceptron_obj = perceptron.Perceptron(eta=0.1, n_iter=10)
    print("Performing perceptron fit..")
    perceptron_obj.fit(x, y)
    # print(perceptron_obj.errors_)

    matplotlib.pyplot.plot(
        range(1, len(perceptron_obj.errors_) + 1),
        perceptron_obj.errors_
    )
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Number of misclassifications")
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


if __name__ == "__main__":
    try:
        test_iris()
    except Exception as error:
        print("ERROR: Iris test error")
        print(str(error))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Line: " + str(exc_tb.tb_lineno))
        exit(1)
