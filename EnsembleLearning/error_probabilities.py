import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp

from utils.common import ensemble_error
from utils.common import heading


def main():

    error_probabilities = np.linspace(0.01, 1.0, 100)
    heading('Error probabilities')
    pp.pprint(error_probabilities)
    print(type(error_probabilities))
    n_classifers = 50
    ensemble_errors = np.array(
        [
            ensemble_error(n_classifers, error_prob)
            for error_prob in error_probabilities
        ]
    )
    heading('Ensemble errors:')
    pp.pprint(ensemble_errors)
    print(type(ensemble_errors))

    plt.plot(
        error_probabilities,
        ensemble_errors,
        label='Ensemble error plot'
    )

    plt.plot(
        error_probabilities,
        error_probabilities,
        label='Base error',
        linestyle=':'
    )

    plt.fill_between(
        error_probabilities,
        ensemble_errors,
        error_probabilities,
        where=(ensemble_errors < error_probabilities),
        alpha=0.25,
        facecolor='green',
        interpolate=True
    )

    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('Error probability per classifier')
    plt.ylabel('Error probability of ensemble')
    plt.show()


if __name__ == '__main__':
    main()
