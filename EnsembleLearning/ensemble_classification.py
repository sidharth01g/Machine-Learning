import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from majority_vote_classifier import MajorityVoteClassifier
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.common import heading
from utils.common import show_error
from utils.datasets_custom import WineExample


def get_data_wine():
    # Load dataset
    wine = WineExample()
    wine.fetch_data()

    X = wine.df.iloc[:, 1:].values
    y = wine.df.iloc[:, 0].values

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.9, random_state=0
    )

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    X_train = X_train[y_train != 2]
    y_train = y_train[y_train != 2]

    X_test = X_test[y_test != 2]
    y_test = y_test[y_test != 2]

    return (X_train, X_test, y_train, y_test)


def get_data_cancer_sklearn():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.5, random_state=1
    )

    return (X_train, X_test, y_train, y_test)


def main():

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(linewidth=200)

    try:
        (X_train, X_test, y_train, y_test) = get_data_cancer_sklearn()
        # (X_train, X_test, y_train, y_test) = get_data_wine()
    except Exception as error:
        show_error('Getting data failed')
        heading('Exception Trace:')
        raise error
    """
    pp.pprint(X_train)
    pp.pprint(X_test)
    pp.pprint(y_train)
    pp.pprint(y_test)
    """

    p1 = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l2',
                C=0.0001,
                random_state=1
            ))
        ]
    )

    p2 = Pipeline(
        [
            ('classifier', DecisionTreeClassifier(
                max_depth=1,
                criterion='entropy',
                random_state=1
            ))
        ]
    )

    classifier_pipes_list = [p1, p2]
    weights = [1, 1]

    maj_vote_clf = MajorityVoteClassifier(
        classifiers=classifier_pipes_list,
        vote='classlabel',
        weights=weights
    )

    maj_vote_clf.fit(X_train, y_train)
    # heading('MajorityVoteClassifier prediction:')
    # pp.pprint(maj_vote_clf.predict(X_test))

    classifiers_list = [p1, p2, maj_vote_clf]

    classifier_labels = [
        'LogisticRegression',
        'DecisionTreeClassifier',
        'MajorityVoteClassifier'
    ]

    for label, clf_pipe in zip(classifier_labels, classifiers_list):
        scores = cross_val_score(
            estimator=clf_pipe,
            X=X_train,
            y=y_train,
            cv=2,
            scoring='roc_auc'
        )
        heading('Classifier: %s' % label)
        print('Mean score: ', np.mean(scores))


if __name__ == '__main__':
    main()
