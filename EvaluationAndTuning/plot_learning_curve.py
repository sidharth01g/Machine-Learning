import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp


from sklearn.decomposition import PCA
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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


def get_fitted_pipeline_lr(X_train, y_train):

    pipeline_lr = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('classifier', LogisticRegression(random_state=1))
        ]
    )

    pipeline_lr.fit(X_train, y_train)
    return pipeline_lr


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

    pipeline_lr = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l2',
                random_state=123
            ))
        ]
    )

    train_sizes = np.linspace(0.1, 1.0, 10)
    (train_sizes, train_scores, test_scores) = learning_curve(
        estimator=pipeline_lr,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes,
        cv=10,
        n_jobs=-1
    )
    heading('train_sizes')
    pp.pprint(train_sizes)
    pp.pprint(train_sizes.shape)
    heading('train_scores')
    pp.pprint(train_scores)
    pp.pprint(train_scores.shape)
    heading('test_scores')
    pp.pprint(test_scores)
    pp.pprint(test_scores.shape)

    train_mean = np.mean(train_scores, axis=1)
    heading('train_mean')
    pp.pprint(train_mean)

    train_std = np.std(train_scores, axis=1)
    heading('train_std')
    pp.pprint(train_std)

    test_mean = np.mean(test_scores, axis=1)
    heading('test_mean')
    pp.pprint(test_mean)

    test_std = np.std(test_scores, axis=1)
    heading('test_std')
    pp.pprint(test_std)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5,
             label='Training accuracy')
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color='blue'
    )

    plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5,
             linestyle='--', label='Validation accuracy')
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color='green'
    )

    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')


    plt.show()


if __name__ == '__main__':
    main()
