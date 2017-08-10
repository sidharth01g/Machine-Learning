import hashlib
import math
import numpy as np
import os
import pandas as pd
import scipy.misc
import shutil
import struct


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

    def fetch_data(self, header=None, sep=None):
        if os.path.exists(self.cache_file_path):
            print("Loading data from cache:", self.cache_file_path)
            self.df = pd.read_pickle(self.cache_file_path)
        else:
            print("Fetching data from %s..\n" % self.dataset_url)
            if sep:
                self.df = pd.read_csv(self.dataset_url, header=header, sep=sep)
            else:
                # Passing sep=None seems to cause an exception. Hence if-else
                # condition
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


def show_error(message):
    return 'ERROR: ' + str(message)


def ensemble_error(n_classifers, error_probability):
    k_start = math.ceil(n_classifers / 2)

    # Probabilites generator
    probabilities = (
        scipy.misc.comb(n_classifers, k)
        * error_probability**k
        * (1.0 - error_probability)**(n_classifers - k)
        for k in range(k_start, n_classifers + 1)
    )

    return sum(probabilities)


def load_mnist_dataset(dir_path, type_='train'):

    images_path = os.path.join(
        dir_path,
        '%s-images-idx3-ubyte' % type_
    )
    labels_path = os.path.join(
        dir_path,
        '%s-labels-idx1-ubyte' % type_
    )

    with open(labels_path, 'rb') as label_file:
        (magic, n) = struct.unpack(
            '>II',
            label_file.read(8)
        )
        labels = np.fromfile(label_file, dtype=np.uint8)

    with open(images_path, 'rb') as image_file:
        (magic, number, rows, columns) = struct.unpack(
            '>IIII',
            image_file.read(16)
        )
        image_data_serial = np.fromfile(image_file, dtype=np.uint8)
        n_images = len(labels)
        n_pixels = 28 * 28
        images = image_data_serial.reshape(
            n_images,
            n_pixels
        )

    return (images, labels)
