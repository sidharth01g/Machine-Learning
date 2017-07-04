import pandas as pd


class WineExample(object):

    def __init__(self):
        self.dataset_url = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/'
            + 'wine/wine.data')
        self.df = None

    def fetch_data(self):
        print("Fetching data from URL..")
        self.df = pd.read_csv(self.dataset_url, header=None)


def test():
    wine = WineExample()
    wine.fetch_data()
    print(wine.df)


if __name__ == '__main__':
    test()
