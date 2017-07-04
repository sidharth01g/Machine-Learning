from io import StringIO
from sklearn.preprocessing import Imputer
import pandas as pd


class Example(object):
    def __init__(self):
        self.csv_string = """A,B,C,D\n1.2,2.6,7.9,3.5\n,72.6,,5.5\n11.2,,,"""

    def get_dataframe(self):
        dataframe = pd.read_csv(StringIO(self.csv_string))
        return dataframe

    def get_nulls(self):
        df = self.get_dataframe()
        return df.isnull()

    def get_null_counts(self):
        df = self.get_dataframe()
        return df.isnull().sum()

    def get_values(self):
        df = self.get_dataframe()
        return df.values

    def get_complete_rows(self):
        df = self.get_dataframe()
        return df.dropna()

    def get_complete_columns(self):
        df = self.get_dataframe()
        return df.dropna(axis=1)

    def get_imputed_data(self, strategy='mean'):
        df = self.get_dataframe()
        imr = Imputer(missing_values='NaN', strategy=strategy, axis=0)
        imr.fit(df)
        imputed_data = imr.transform(df.values)
        return imputed_data


def run_example():
    ex = Example()
    df = ex.get_dataframe()

    print("Example dataframe:")
    print(df, '\n')

    print("Printing values as underlying array:")
    values_ = ex.get_values()
    print(values_, "\n")

    print("Mean strategy interpolated data:")
    imputed_data = ex.get_imputed_data()
    print(imputed_data, '\n')

    print("Median strategy interpolated data:")
    imputed_data = ex.get_imputed_data(strategy='median')
    print(imputed_data, '\n')

    print('Most-frequent strategy interpolated data:')
    imputed_data = ex.get_imputed_data(strategy='most_frequent')
    print(imputed_data, '\n')


if __name__ == "__main__":
    run_example()
