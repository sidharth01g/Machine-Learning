from datasets_custom import WineExample
import matplotlib.pyplot as plt
import numpy as np
import plots
import pprint as pp
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def heading(text, character='='):
    if type(text) is not str:
        heading('<INVALID_HEADING>')
        return
    print('\n')
    print(text)
    print(character * len(text))


def test():
    np.set_printoptions(suppress=True)
    wine = WineExample()
    wine.fetch_data()

    X = wine.df.iloc[:, 1:].values
    y = wine.df.iloc[:, 0].values

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Scale the feature vectors
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    pp.pprint(X_train_lda)

    lr = LogisticRegression()
    lr.fit(X_train_lda, y_train)

    custom_plot = plots.CustomPlots()
    custom_plot.plot_decision_regions_1(X_train_lda, y_train, classifier=lr)

    plt.xlabel('Linear discriminant 1')
    plt.ylabel('Linear discriminant 2')
    plt.show()


if __name__ == '__main__':
    test()
