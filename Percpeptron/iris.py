import pandas
import perceptron
import matplotlib.pyplot
import numpy


def get_dataframe(url, header=None):

    try:
        dataframe_obj = pandas.read_csv(url, header)
    except Exception as error:
        print("Exception occured while getting dataframe")
        print(str(error))
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
    matplotlib.pyplot.show()


if __name__ == "__main__":
    try:
        test_iris()
    except Exception as error:
        print("ERROR: Iris test error")
        exit(1)
