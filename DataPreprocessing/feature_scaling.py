from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


def test():
    # Load dataset
    iris = datasets.load_iris()

    X = iris.data[:, [0, 1]]
    y = iris.target[:]

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    print(X_train_norm)

    ss = StandardScaler()
    X_train_std = ss.fit_transform(X_train)
    print(X_train_std)

    print(X_train_norm - X_train_std)


if __name__ == '__main__':
    test()
