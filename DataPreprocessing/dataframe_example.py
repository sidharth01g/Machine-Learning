from io import StringIO
import pandas as pd


class Example(object):
    def __init__(self):
        self.csv_string = """A,B,C,D
        1.2,2.6,7.9,3.5
        ,72.6,,5.5
        11.2,,,"""

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


def run_example():
    ex = Example()
    df = ex.get_dataframe()

    print("Example dataframe:")
    print(df, '\n')

    print("Printing null values:")
    nulls = ex.get_nulls()
    print(nulls, '\n')

    print("Printing null counts:")
    null_counts = ex.get_null_counts()
    print(null_counts, '\n')

    print("Printing values as underlying array:")
    values_ = ex.get_values()
    print(values_, "\n")

    print("Printing complete rows only:")
    comp_rows = ex.get_complete_rows()
    print(comp_rows)



if __name__ == "__main__":
    run_example()
