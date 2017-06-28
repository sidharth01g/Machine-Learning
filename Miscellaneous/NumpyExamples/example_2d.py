import numpy as np

def f(x, y):
    return x**3 + y**2

a = np.fromfunction(f, (10, 10), dtype=int)
print(a)
print(a)
