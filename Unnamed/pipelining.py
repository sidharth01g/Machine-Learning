# This program is not performing well. Need to debug

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import numpy as np
import pprint as pp

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import utils.common
from utils.common import heading
from utils.common import show_error


def interpolate(features, replacement, strategy='mean'):
    features[features == replacement[0]] = np.nan
    imputer = Imputer(missing_values=replacement[1], strategy=strategy, axis=0)
    imputer.fit(features)
    return imputer.transform(features)


def get_data():
    # URL for Wisconsin breast cancer dataset
    url = (
        'https://archive.ics.uci.edu/ml/machine-learning-databases/'
        + 'breast-cancer-wisconsin/wpbc.data'
    )

    rl = utils.common.RemoteDataLoader(url)
    try:
        rl.fetch_data()
    except Exception as error:
        message = 'ERROR: Exception while fetching remote data'
        print(message)
        raise error

    df = rl.get_dataframe()

    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.50, random_state=1
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

    # Replace missing features '?' with 'non a number'
    replacement = ('?', np.nan)

    try:
        X_train = interpolate(X_train, replacement)
    except Exception as error:
        show_error('Interpolating X-train failed')
        heading('Exception Trace:')
        raise error

    heading('Training vectors (interpolated):')
    pp.pprint(X_train)
    print('Shape: ', X_train.shape)


    try:
        pipeline_lr = get_fitted_pipeline_lr(X_train, y_train)
    except Exception as error:
        show_error('Fitting pipeline failed')
        heading('Exception Trace:')
        raise error

    heading('Test vectors:')
    pp.pprint(X_test)
    print('Shape: ', X_test.shape)
    try:
        X_test = interpolate(X_test, replacement)
    except Exception as error:
        show_error('Interpolating X-test failed')
        heading('Exception Trace:')
        raise error

    heading('Test vectors (interpolated):')
    pp.pprint(X_test)
    print('Shape: ', X_test.shape)

    print(y_test)
    print(pipeline_lr.predict(X_test))

    message = 'Accuracy: %s' % pipeline_lr.score(X_test, y_test)
    print(message)


if __name__ == '__main__':
    main()
