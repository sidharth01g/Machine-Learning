import hashlib
import os
import pandas as pd
import shutil


class RemoteDataLoader(object):

    def __init__(self, dataset_url):
        if type(dataset_url) is not str:
            raise TypeError('URL not a string')

        self.dataset_url = dataset_url
        self.df = None

        # Program path
        self.program_path = os.path.abspath(__file__)
        self.program_dir_path = os.path.dirname(self.program_path)

        # Create hash values for URL. Generate cache file names based on the
        # URL hash values
        hash_obj = hashlib.sha1(self.dataset_url.encode('utf-8'))
        hash_value = hash_obj.hexdigest()

        self.local_cache_foldername = '.cache'
        self.cache_filename = 'data_cache_%s.pkl' % hash_value
        self.cache_dir_path = os.path.abspath(
            self.program_dir_path + '/' + self.local_cache_foldername
        )
        self.cache_file_path = os.path.abspath(
            self.cache_dir_path + '/' + self.cache_filename
        )

    def fetch_data(self, header=None):
        if os.path.exists(self.cache_file_path):
            print("Loading data from cache:", self.cache_file_path)
            self.df = pd.read_pickle(self.cache_file_path)
        else:
            print("Fetching data from %s..\n" % self.dataset_url)
            self.df = pd.read_csv(self.dataset_url, header=header)

            if not os.path.exists(self.cache_dir_path):
                print("Creating directory:", self.cache_dir_path)
                os.makedirs(self.cache_dir_path)

                print("Creating cache:", self.cache_file_path)
                self.df.to_pickle(self.cache_file_path)

    def clear_cache(self):
        if os.path.exists(self.cache_dir_path):
            print("Removing cache folder:", self.cache_dir_path)
            shutil.rmtree(self.cache_dir_path)
        else:
            print("Warning: Cache already empty")

    def get_dataframe(self):
        return self.df


def heading(text, character='='):
    if type(text) is not str:
        heading('<INVALID_HEADING>')
        return
    print('\n')
    print(text)
    print(character * len(text))
