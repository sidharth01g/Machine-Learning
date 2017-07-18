import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import numpy as np
import pprint as pp


from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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

    try:
        pipeline_lr = get_fitted_pipeline_lr(X_train, y_train)
    except Exception as error:
        show_error('Fitting pipeline failed')
        heading('Exception Trace:')
        raise error

    kfold = StratifiedKFold(
        y=y_train,
        n_folds=10,
        random_state=123
    )

    heading('KFold:')
    pp.pprint(kfold)

    scores = []

    for index, (train, test) in enumerate(kfold):
        heading('Fold: %s' % str(index))
        print('Training set:')
        pp.pprint(train)
        print('Testing set:')
        pp.pprint(test)
        pipeline_lr.fit(X_train[train], y_train[train])
        score = pipeline_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print(
            'Class dis %s\n' % np.bincount(y_train[train])
            + 'Score: %.3f' % score
        )


if __name__ == '__main__':
    main()
