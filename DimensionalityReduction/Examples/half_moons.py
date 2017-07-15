import matplotlib.pyplot as plt
import pprint as pp
from sklearn.datasets import make_moons
from utils.common_methods import heading


def test():

    (X, y) = make_moons(n_samples=100, random_state=123)

    heading('Feature vectors: ')
    pp.pprint(X)
    heading('Labels: ')
    pp.pprint(y)

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color='red',
        marker='^',
        alpha=0.5
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='blue',
        marker='o',
        alpha=0.5
    )

    plt.show()


if __name__ == '__main__':
    test()
