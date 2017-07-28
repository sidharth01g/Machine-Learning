import numpy as np
import os
import pandas as pd
import pprint as pp
import pyprind
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from utils.common import heading


class MovieData(object):

    def __init__(self, n_reviews):
        self.n_reviews = n_reviews
        self.dataset_categories = ['test', 'train']
        self.root_dirname = 'aclImdb'
        self.labels_dict = {
            'pos': 1,
            'neg': 0
        }
        self.csv_filename = 'reviews.csv'

    def read_data(self):

        progress_bar = pyprind.ProgBar(self.n_reviews)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        csv_filepath = os.path.join(current_dir, self.csv_filename)
        if os.path.exists(csv_filepath):
            print('Reading dataset from ', csv_filepath, '..')
            df = pd.read_csv(csv_filepath)
        else:

            df = pd.DataFrame()

            for dataset in self.dataset_categories:
                for label in self.labels_dict.keys():
                    path = os.path.abspath(
                        current_dir + '/' + 'movie_review_dataset' + '/'
                        + self.root_dirname + '/' + dataset + '/' + label)
                    if not os.path.exists(path):
                        raise Exception(path + ' does not exist')

                    for file_ in os.listdir(path):
                        file_path = os.path.join(path, file_)
                        with open(file_path, 'r') as infile:
                            text = infile.read()
                        df = df.append(
                            [
                                [
                                    text,
                                    self.labels_dict[label]
                                ]

                            ],
                            ignore_index=True
                        )
                        progress_bar.update()
            print('Writing to file ', csv_filepath, '..')
            df.to_csv(csv_filepath, index=False)

        df.columns = ['review', 'sentiment']
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))

        self.df = df
        return df


def main():
    movies = MovieData(n_reviews=50000)
    print('Reading data..')
    movies.read_data()
    heading('Data:')
    pp.pprint(movies.df)


if __name__ == '__main__':
    main()
