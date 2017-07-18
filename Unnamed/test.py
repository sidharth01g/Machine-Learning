import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
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


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
pp.pprint(X)
pp.pprint(y)

print(le.transform(['R', 'N']))

(X_train, X_test, y_train, y_test) = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1
)

X_train[X_train == '?'] = np.nan
X_test[X_test == '?'] = np.nan


imr = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imr.fit(X_train)
X_train = imr.transform(X_train)
X_test = imr.transform(X_test)
print('=' * 200)
pp.pprint(X_train)
pp.pprint(X_test)
pp.pprint(y_train)
pp.pprint(y_test)


pipeline_lr = Pipeline(
    [
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(random_state=1))
    ]
)
pipeline_lr.fit(X_train, y_train)
pp.pprint(pipeline_lr.predict(X_test))
message = 'Accuracy: %s' % pipeline_lr.score(X_test, y_test)
print(message)
