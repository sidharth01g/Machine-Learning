import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from utils.common import heading
from utils.common import RemoteDataLoader


def run():
    data_x = [
        258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0
    ]
    data_y = [
        236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8
    ]

    # Convert data lists numpy arrays
    X = np.array(data_x)
    y = np.array(data_y)

    # Make shape of X (n_samples, n_features):
    # standard form for learning models
    X = X[:, np.newaxis]

    heading('Collecting data')
    url = (
        'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/'
        + 'housing.data'
    )
    print('Reading: ', url)
    rdl = RemoteDataLoader(url)
    rdl.fetch_data(sep='\s+')  # separator: one or more whitespaces ('\s+')
    df = rdl.df

    # Name the columns
    pp.pprint(df.values.shape)
    df.columns = [
        'CRIM',
        'ZN',
        'INDUS',
        'CHAS',
        'NOX',
        'RM',
        'AGE',
        'DIS',
        'RAD',
        'TAX',
        'PTRATIO',
        'B',
        'LSTAT',
        'MEDV'
    ]
    heading('Listing a few samples')
    pp.pprint(df.head())

    X = df[['RM']].values
    y = df[['MEDV']].values

    # Create a polynomial feature transformer
    # x -->   < 1  x  x^2  x^3  ...  x^(degree - 1) >
    degree = 5
    poly_transformer = PolynomialFeatures(degree=degree)
    X_poly = poly_transformer.fit_transform(X)

    heading('Data:')
    print('\nX:')
    pp.pprint(X)
    pp.pprint(X.shape)
    print('\ny:')
    pp.pprint(y)
    pp.pprint(y.shape)

    heading('X - polynomial representaion, degree=%s' % degree)
    pp.pprint(X_poly)
    pp.pprint(X_poly.shape)

    # LinearRegression instances
    lr_linear = LinearRegression()
    lr_poly = LinearRegression()

    print('\nFitting models.. ')
    print('1. Linear regression model')
    lr_linear.fit(X, y)
    print('2. Polynomial regression model')
    lr_poly.fit(X_poly, y)

    # Create data to run prediction on
    X_test = np.linspace(
        start=int(min(X)),
        stop=int(max(X)),
        num=100
    )
    X_test = X_test[:, np.newaxis]
    X_test_poly = poly_transformer.fit_transform(X_test)

    heading('Predictions:')
    y_predict_linear = lr_linear.predict(X_test)
    y_predict_poly = lr_poly.predict(X_test_poly)
    pp.pprint(y_predict_linear)
    pp.pprint(y_predict_poly)

    heading('Metrics:')
    print(
        'Mean squared error (linear): ',
        mean_squared_error(
            y, lr_linear.predict(X)
        )
    )
    print(
        'Mean squared error (poly): ',
        mean_squared_error(
            y, lr_poly.predict(X_poly)
        )
    )
    print(
        'R2 score (linear): ',
        r2_score(
            y, lr_linear.predict(X)
        )
    )
    print(
        'R2 score (poly): ',
        r2_score(
            y, lr_poly.predict(X_poly)
        )
    )

    plt.figure()
    plt.scatter(X, y, label='Training points')
    plt.plot(X_test, y_predict_linear, label='Linear fit')
    plt.plot(X_test, y_predict_poly, label='Polynomial fit')
    plt.legend(loc='best')
    plt.xlabel('Predictor')
    plt.ylabel('Response')
    plt.title('Polynomial degree = %s' % degree)

    plt.show()


if __name__ == '__main__':
    run()
