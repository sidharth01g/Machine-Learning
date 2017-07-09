from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def test():
    # Load dataset
    iris = datasets.load_iris()

    X = iris.data[:, [1, 3]]
    y = iris.target[:]

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    ss = StandardScaler()
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)

    # Use logistic regression with a regularization parameter C=1/lambda
    c_generator = (10**i for i in range(-3, 3))
    print(type(c_generator))

    for c in c_generator:
        lr = LogisticRegression(penalty='l1', C=c)
        lr.fit(X_train_std, y_train)

        train_set_accuracy = lr.score(X_train_std, y_train)
        test_set_accuracy = lr.score(X_test_std, y_test)
        print("C = %s, lambda = %s" % (str(c), str(1.0/c)))
        print("Training accuracy: ", train_set_accuracy)
        print("Test accuracy:", test_set_accuracy, '\n\n')


if __name__ == '__main__':
    test()
