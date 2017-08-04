import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import pprint as pp
from utils.common import RemoteDataLoader



def run():
    url = (
        'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/'
        + 'housing.data'
    )
    print('Reading: ', url)
    rdl = RemoteDataLoader(url)
    rdl.fetch_data()
    pp.pprint(rdl.df)


if __name__ == '__main__':
    run()
