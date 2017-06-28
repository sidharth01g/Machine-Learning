import numpy as np

def f(x, y):
    return x**2 * y**2

a = np.fromfunction(f, (10, 10), dtype=int)
print(a)
