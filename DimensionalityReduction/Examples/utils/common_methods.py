import numpy as np


def heading(text, character='='):
    if type(text) is not str:
        heading('<INVALID_HEADING>')
        return
    print('\n')
    print(text)
    print(character * len(text))


def project_x(X_sample, X, gamma, alphas, lambdas):
    pair_distance = np.array(
        [np.sum(
            (X_sample - row)**2 for row in X)]
    )
    k = np.exp(-gamma * pair_distance)
    return k.dot(alphas / lambdas)
