import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import pprint as pp
import utils.common
from utils.common import heading


def test():
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

    heading('Dataframe:')
    pp.pprint(df)

    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values

    heading('Feature vectors:')
    pp.pprint(X)
    print('Shape: ', X.shape)

    heading('Class labels:')
    pp.pprint(y)
    print('Shape: ', y.shape)





if __name__ == '__main__':
    try:
        test()
    except Exception as error:
        heading('Exception Trace:')
        raise error
