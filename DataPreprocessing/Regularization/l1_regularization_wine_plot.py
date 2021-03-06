import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class WineExample(object):

    def __init__(self):
        self.dataset_url = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/'
            + 'wine/wine.data')
        self.df = None
        self.class_label_string = 'Class label'

        # Program path
        self.program_path = os.path.abspath(__file__)
        self.program_dir_path = os.path.dirname(self.program_path)

        # Cache path variables
        self.local_cache_foldername = '.cache'
        self.cache_filename = 'wine.pkl'
        self.cache_dir_path = os.path.abspath(
            self.program_dir_path + '/' + self.local_cache_foldername
        )
        self.cache_file_path = os.path.abspath(
            self.cache_dir_path + '/' + self.cache_filename
        )

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
        else:
            print("Warning: Cache already empty")


def test():
    wine = WineExample()
    wine.fetch_data()
    print('Class labels: ', np.unique(wine.df[wine.class_label_string]), '\n')

    X = wine.df.iloc[:, 1:].values
    y = wine.df.iloc[:, 0].values

    # Partition into training and testing samples
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    print("Training set shape:", X_train.shape)
    print("Tesing set shape:", X_test.shape, '\n')

    # Standardization
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)

    weights = []
    params = []

    c_generator = (10**i for i in range(-3, 3))
    class_index = 2

    for c in c_generator:
        lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[class_index])
        params.append(10**c)

    weights = np.array(weights)
    print(weights)

    plt.figure()
    plt.subplot(111)

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
              'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

    for column, color in zip(range(weights.shape[1]), colors):
        feature_name = wine.df.columns[column + 1]
        plt.plot(params, weights[:, column], label=feature_name, color=color)

    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**-5, 10**5])
    plt.ylabel("Weight coefficients")
    plt.xlabel("Value of C")
    plt.xscale('log')
    plt.legend(
        loc='upper left', ncol=1, fancybox=True
    )
    plt.show()






if __name__ == '__main__':
    test()
