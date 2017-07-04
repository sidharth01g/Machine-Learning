import numpy as np
import os
import pandas as pd
import shutil


class WineExample(object):

    def __init__(self):
        self.dataset_url = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/'
            + 'wine/wine.data')
        self.df = None

        self.local_cache_foldername = '.cache'
        self.cache_filename = 'wine.pkl'

        self.program_path = os.path.abspath(__file__)
        self.program_dir_path = os.path.dirname(self.program_path)
        self.cache_dir_path = os.path.abspath(
            self.program_dir_path + '/' + self.local_cache_foldername
        )
        self.cache_file_path = os.path.abspath(
            self.cache_dir_path + '/' + self.cache_filename
        )

        self.class_label_string = 'Class label'

    def fetch_data(self):
        if os.path.exists(self.cache_file_path):
            print("Loading data from cache:", self.cache_file_path)
            self.df = pd.read_pickle(self.cache_file_path)
        else:
            print("Fetching data from %s..\n" % self.dataset_url)
            self.df = pd.read_csv(self.dataset_url, header=None)
            self.df.columns = [
                self.class_label_string,
                'Alcohol',
                'Malic acid',
                'Ash',
                'Alcalinity of ash',
                'Magnesium',
                'Total phenols',
                'Falavanoids',
                'Nonflavanoid phenols',
                'Proanthocyanins',
                'Color instensity',
                'Hue',
                'OD280/OD315 of diluted wines',
                'Proline'
            ]

            if not os.path.exists(self.cache_dir_path):
                print("Creating directory:", self.cache_dir_path)
                os.makedirs(self.cache_dir_path)

                print("Creating cache:", self.cache_file_path)
                self.df.to_pickle(self.cache_file_path)

    def clear_cache(self):
        if os.path.exists(self.cache_dir_path):
            print("Removing cache folder:", self.cache_dir_path)
            shutil.rmtree(self.cache_dir_path)



def test():
    wine = WineExample()
    wine.fetch_data()
    print('Class labels: ', np.unique(wine.df[wine.class_label_string]), '\n')
    # wine.clear_cache()


if __name__ == '__main__':
    test()
