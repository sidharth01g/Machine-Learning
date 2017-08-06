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
from sklearn.preprocessing import PolynomialFeatures
from utils.common import heading


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

    # Create a polynomial feature transformer
    # x -->   < 1  x  x^2  x^3  ...  x^(degree - 1) >

    # Compute mean_squared_error on using training data
    stats = []
    degrees = [i for i in range(2, 11)]
    for degree in degrees:
        poly_transformer = PolynomialFeatures(degree=degree)
        X_poly = poly_transformer.fit_transform(X)

        # LinearRegression instances
        lr_linear = LinearRegression()
        lr_poly = LinearRegression()

        print('\nFitting models.. ')
        print('1. Linear regression model')
        lr_linear.fit(X, y)
        print('2. Polynomial regression model')
        lr_poly.fit(X_poly, y)

        # Save stats
        stats.append(
            {
                'linear': {
                    'mean squared error': mean_squared_error(
                        y, lr_linear.predict(X))
                },
                'poly': {
                    'mean squared error': mean_squared_error(
                        y, lr_poly.predict(X_poly))
                }
            }
        )
    mse_linear = np.array(
        [
            x['linear']['mean squared error']
            for x in stats
        ]
    )
    mse_poly = np.array(
        [
            x['poly']['mean squared error']
            for x in stats
        ]
    )

    print('==' * 50)
    pp.pprint(stats)

    plt.figure()
    plt.plot(np.array(degrees), mse_linear)
    plt.plot(np.array(degrees), mse_poly)
    plt.show()



if __name__ == '__main__':
    run()
