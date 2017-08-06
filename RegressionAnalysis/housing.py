import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from utils.common import heading
from utils.common import RemoteDataLoader


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')


def run():
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

    # Pairwise scatter plot for a few features
    cols = df.columns[10:]
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(
        df[cols],
    )
    plt.title('Pairwise scatter plots for a few features')
    plt.figure()

    heading('Correlation matrix for: %s' % cols)
    feature_samples = df[cols].values.T
    correlation_matrix = np.corrcoef(feature_samples)
    pp.pprint(correlation_matrix)
    sns.heatmap(
        correlation_matrix,
        cbar=True,
        annot=True,
        square=True,
        fmt='2f',
        annot_kws={'size': 15},
        yticklabels=cols,
        xticklabels=cols
    )
    plt.title('Heatmap for scatter matrix')
    # plt.show()

    heading('Linear Regression')
    X = df[['RM']].values
    y = df[['MEDV']].values

    X_std = StandardScaler().fit_transform(X)
    y_std = StandardScaler().fit_transform(y)

    print('X_std: ', X_std.shape)
    print('y_std', y_std.shape)

    lr = LinearRegression()
    lr.fit(X, y)

    print('\nCoefficients:')
    pp.pprint(lr.coef_)
    print('Intercept:')
    pp.pprint(lr.intercept_)
    plt.figure()
    lin_regplot(X, y, lr)
    plt.xlabel('Average number of rooms (RM)')
    plt.ylabel('Price in $1000s (MEDV)')
    plt.title('sklearn.linear_model.LinearRegression')

    heading('RANSAC')
    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=50,
        residual_metric=lambda x: np.sum(np.abs(x), axis=1),
        residual_threshold=5.0,
        random_state=0
    )
    ransac.fit(X, y)

    # Create masks
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    print('\nInlier mask: ', inlier_mask.shape)
    print('Outlier mask: ', outlier_mask.shape)

    plt.figure()
    plt.title('RANSAC: Inliers and outliers')
    plt.xlabel('Average number of rooms (RM)')
    plt.ylabel('Price in $1000s (MEDV)')
    plt.scatter(
        X[inlier_mask],
        y[inlier_mask],
        c='blue',
        marker='o',
        label='Inliers'
    )

    plt.scatter(
        X[outlier_mask],
        y[outlier_mask],
        c='red',
        marker='o',
        label='Outliers'
    )
    plt.legend(loc='best')

    X_max = max(X)
    X_min = min(X)
    regression_line_X = np.arange(X_min, X_max + 1, 1)
    regression_line_y = ransac.predict(regression_line_X[:, np.newaxis])
    plt.plot(
        regression_line_X,
        regression_line_y,
        color='black',
    )
    print('\nCoefficients:')
    pp.pprint(ransac.estimator_.coef_)
    print('Intercept')
    pp.pprint(ransac.estimator_.intercept_)
    plt.show()


if __name__ == '__main__':
    run()
