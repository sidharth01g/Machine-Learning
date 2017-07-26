import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from utils.common import heading
from utils.datasets_custom import WineExample


def get_data():
    # Load dataset
    wine = WineExample()
    wine.fetch_data()

    X = wine.df.iloc[:, 1:].values
    y = wine.df.iloc[:, 0].values

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    return (X_train, X_test, y_train, y_test)


def main():
    (X_train, X_test, y_train, y_test) = get_data()

    # Get rid of one of the 3 classes to convert to binary calssification
    X_train = X_train[y_train != 0][:, 0: 2]
    y_train = y_train[y_train != 0]
    X_test = X_test[y_test != 0][:, 0: 2]
    y_test = y_test[y_test != 0]

    print(y_train)
    print(y_test)

    dtc = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=None
    )

    n_estimators = 500

    # The number of samples to draw from X to train each base estimator
    max_samples = 1.0

    # The number of features to draw from X to train each base estimator
    max_features = 1.0

    # bootstrap: Whether samples are drawn with replacement
    bag = BaggingClassifier(
        base_estimator=dtc,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,
        bootstrap_features=False,
    )

    # Accuracy of single decision tree classifier
    heading('Single decision tree classifier')
    dtc = dtc.fit(X_train, y_train)
    y_predict_dtc = dtc.predict(X_test)
    score_dtc = accuracy_score(y_test, y_predict_dtc)
    print('Score: ', score_dtc)

    # Accuracy with bagging
    heading('Bagging classifier')
    bag = bag.fit(X_train, y_train)
    y_predict_bag = bag.predict(X_test)
    score_bag = accuracy_score(y_test, y_predict_bag)
    print('Score: ', score_bag)


if __name__ == '__main__':
    main()
