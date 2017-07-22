import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import numpy as np
import pprint as pp


from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.common import heading
from utils.common import show_error
from utils.datasets_custom import WineExample


def interpolate(features, replacement, strategy='mean'):
    features[features == replacement[0]] = np.nan
    imputer = Imputer(missing_values=replacement[1], strategy=strategy, axis=0)
    imputer.fit(features)
    return imputer.transform(features)


def get_data():
    # Load dataset
    wine = WineExample()
    wine.fetch_data()

    X = wine.df.iloc[:, 1:].values
    y = wine.df.iloc[:, 0].values

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    return (X_train, X_test, y_train, y_test)


def main():

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(linewidth=200)

    try:
        (X_train, X_test, y_train, y_test) = get_data()
    except Exception as error:
        show_error('Getting data failed')
        heading('Exception Trace:')
        raise error

    heading('Training vectors:')
    pp.pprint(X_train)
    print('Shape: ', X_train.shape)

    heading('Training class labels:')
    pp.pprint(y_train)
    print('Shape: ', y_train.shape)

    # Create separate pipelines for Support Vector Machine and Decision Tree
    pipeline_svm = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('classifier', SVC(random_state=1))
        ]
    )

    pipeline_decision_tree = Pipeline(
        [
            ('classifier', DecisionTreeClassifier(random_state=1))
        ]
    )
    heading('SVM pipeline:')
    pp.pprint(pipeline_svm)
    heading('Decision tree pipeline:')
    pp.pprint(pipeline_decision_tree)

    # Create separate parameter grids for
    # Support Vector Machine and Decision Tree

    parameter_grid_svm = [
        {
            'classifier__kernel': ['linear'],
            'classifier__C': [10**i for i in range(-5, 5)]
        },
        {
            'classifier__kernel': ['rbf'],
            'classifier__C': [10**i for i in range(-5, 5)],
            'classifier__gamma': [10**i for i in range(-5, 5)]
        },
    ]

    parameter_grid_decision_tree = [
        {
            'classifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None]
        }
    ]

    # Create Grid Searches for cross validation

    k_inner = 5

    grid_search_svm = GridSearchCV(
        estimator=pipeline_svm,
        param_grid=parameter_grid_svm,
        scoring='accuracy',
        cv=k_inner,
        n_jobs=1
    )

    grid_search_decision_tree = GridSearchCV(
        estimator=pipeline_decision_tree,
        param_grid=parameter_grid_decision_tree,
        scoring='accuracy',
        cv=k_inner,
        n_jobs=1
    )

    heading(
        'SVM grid_search on  X_train WITHOUT NESTED '
        + 'k-fold cross validation '
        + '(k = %s)' % str(k_inner))
    # pp.pprint(grid_search_svm)
    grid_search_svm.fit(X_train, y_train)
    print('Best score: ', grid_search_svm.best_score_)
    print('Best choice of paramters: ', grid_search_svm.best_params_)
    best_classifier_svm = grid_search_svm.best_estimator_
    best_classifier_svm.fit(X_train, y_train)
    print('Score on test set: ', best_classifier_svm.score(X_test, y_test))

    heading(
        'Decision Tree grid_search on X_train WITHOUT NESTED '
        + 'k-fold cross validation '
        + '(k = %s)' % str(k_inner))
    # pp.pprint(grid_search_decision_tree)
    grid_search_decision_tree.fit(X_train, y_train)
    print('Best score: ', grid_search_decision_tree.best_score_)
    print(
        'Best choice of paramters: ',
        grid_search_decision_tree.best_params_)
    best_classifier_decision_tree = grid_search_decision_tree.best_estimator_
    best_classifier_decision_tree.fit(X_train, y_train)
    print(
        'Score on test set: ',
        best_classifier_decision_tree.score(X_test, y_test))

    # Evaluate scores using each algorithm - SVM and Decision Tree

    k_outer = 10

    scores_svm = cross_val_score(
        grid_search_svm,
        X_train,
        y_train,
        scoring='accuracy',
        cv=k_outer,
        n_jobs=1
    )

    heading(
        'SVM grid_search on  X_train WITH NESTED k-fold cross validation '
        + '(k_outer = %s, k_inner = %s)' % (str(k_outer), str(k_inner)))
    pp.pprint(scores_svm)
    pp.pprint(scores_svm.shape)

    scores_decision_tree = cross_val_score(
        grid_search_decision_tree,
        X_train,
        y_train,
        scoring='accuracy',
        cv=k_outer,
        n_jobs=1
    )

    heading(
        'Decision Tree grid_search on  X_train WITH NESTED '
        + 'k-fold cross validation '
        + '(k_outer = %s, k_inner = %s)' % (str(k_outer), str(k_inner)))
    pp.pprint(scores_decision_tree)
    pp.pprint(scores_decision_tree.shape)

    # Compute mean cross_validation scores
    mean_score_svm = np.mean(scores_svm)
    mean_score_decision_tree = np.mean(scores_decision_tree)

    heading('Mean sores:')
    print('Support Vector Machine: ', mean_score_svm)
    print('Decision Tree:', mean_score_decision_tree)
    heading('')


if __name__ == '__main__':
    main()
