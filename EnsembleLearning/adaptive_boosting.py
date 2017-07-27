import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
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

    dt = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=1
    )

    adaboost = AdaBoostClassifier(
        base_estimator=dt,
        n_estimators=1000,
        learning_rate=0.2,
        random_state=1
    )

    print('Fitting classifiers..')
    # Fit classifiers
    dt.fit(X_train, y_train)
    adaboost.fit(X_train, y_train)

    # Predict on training set
    y_predict_train_dt = dt.predict(X_train)
    y_predict_train_adaboost = adaboost.predict(X_train)

    # Predict on test set
    y_predict_test_dt = dt.predict(X_test)
    y_predict_test_adaboost = adaboost.predict(X_test)

    score_y_predict_train_dt = accuracy_score(y_train, y_predict_train_dt)
    score_y_predict_train_adaboost = accuracy_score(
        y_train, y_predict_train_adaboost)

    score_y_predict_test_dt = accuracy_score(y_test, y_predict_test_dt)
    score_y_predict_test_adaboost = accuracy_score(
        y_test, y_predict_test_adaboost)

    heading('Scores:')
    print('\nScores on training set:\n')
    print('\tDecision Tree: ', score_y_predict_train_dt)
    print('\tAdaBoost (Decision Tree): ', score_y_predict_train_adaboost)

    print('\nScores on test set:\n')
    print('\tDecision Tree: ', score_y_predict_test_dt)
    print('\tAdaBoost (Decision Tree): ', score_y_predict_test_adaboost)


if __name__ == '__main__':
    main()
