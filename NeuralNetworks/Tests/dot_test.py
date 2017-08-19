import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
top_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, top_dir)

import numpy as np
import time
from utils.common import heading

dim = 1000000
w = np.random.rand(dim)
x = np.random.rand(dim)

heading('Sizes:')
print('w: ', w.shape)
print('x: ', x.shape)

heading('Vectorized')
t_start = time.time()
result = np.dot(w, x)
t_stop = time.time()
t_ms = (t_stop - t_start) * 1000
print('Result: ', result.shape)
print('Time: ', t_ms, ' milliseconds')

heading('Non-vectorized')
t_start = time.time()
result = 0
for i in range(w.shape[0]):
    result += w[i] * x[i]
t_stop = time.time()
t_ms = (t_stop - t_start) * 1000
print('Result: ', result.shape)
print('Time: ', t_ms, ' milliseconds')
