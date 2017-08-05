import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import seaborn as sns
from utils.common import RemoteDataLoader
from utils.common import heading


def run():
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
    #plt.show()

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
    plt.show()


if __name__ == '__main__':
    run()
