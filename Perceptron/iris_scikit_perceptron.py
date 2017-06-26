from sklearn import datasets
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


def get_mean(x):
    return (sum(x)/len(x))


def get_variance(x):
    mean = get_mean(x)
    diff = x - mean
    norm_square = [(numpy.linalg.norm(d)) ** 2 for d in diff]
    return (sum(norm_square)/len(x))


def test():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target

    # Split dataset into training and test samples in the ratio
    # test_size : (1- test_size)
    (x_train, x_test, y_train, y_test) = train_test_split(
        x, y, test_size=0.3, random_state=0)

    # Scale feature vectors: Make zero mean and unit variance along each
    # dimension
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    print("\nTraining sample means:")
    print("Non-standardized: ", get_mean(x_train))
    print("Standardized:", get_mean(x_train_std))

    print("\nTraining sample variances:")
    print("Non-standardized: ", get_variance(x_train))
    print("Standardized:", get_variance(x_train_std))

    print("\nScaler fit mean, variance:")
    print(scaler.mean_)
    print("(X-variance, Y-variance): ", scaler.var_)
    print("X-variance + Y-variance: ", sum(scaler.var_))

    print("\nTraining perceptron..")
    ppn = Perceptron(n_iter=80, eta0=1, random_state=0)
    ppn.fit(x_train_std, y_train)

    print("Testing perceptron..")
    y_predicted = ppn.predict(x_test_std)
    print("Predicted:  ", y_predicted)
    print("Actual:     ", y_test)
    print("Number of misclassifications: ",
          (y_predicted != y_test).sum())


if __name__ == "__main__":
    test()
