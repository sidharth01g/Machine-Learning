import matplotlib.pyplot as plt
import numpy as np
import pprint as pp


def run():
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    y_prime = np.exp(-x) / ((1 + np.exp(-x)) ** 2)
    pp.pprint(x)
    pp.pprint(y)
    (fig, ax) = plt.subplots(2, 1, sharex=True)

    ax[0].grid()
    ax[0].plot(x, y)

    ax[1].grid()
    ax[1].plot(x, y_prime)

    plt.show()

if __name__ == '__main__':
    run()
