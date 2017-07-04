import pandas as pd


class Example(object):
    def __init__(self):
        self.columns = ['Color', 'Size', 'Price', 'Class Label']
        self.data = [
            ['Green', 'M', 10.5, 'Class 1'],
            ['Red', 'L', 14.0, 'Class 2'],
            ['Blue', 'XL', 15.6, 'Class 1'],
        ]

    def get_dataframe(self):
        df = pd.DataFrame(self.data)
        df.columns = self.columns
        return df


def run():
    ex = Example()

    print("T-shirts dataframe:")
    df = ex.get_dataframe()
    print(df, '\n')


if __name__ == '__main__':
    run()
