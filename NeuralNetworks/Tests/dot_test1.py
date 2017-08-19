import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
top_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, top_dir)

import numpy as np
import time
from utils.common import heading


m = 1000
n = 100000

heading('Input')
A = np.random.rand(m, n)
v = np.random.rand(n)
print('A: ', A.shape)
print('v: ', v.shape)

heading('Vectorized multiplication')
t_start = time.time()
u = np.dot(A, v)
t_stop = time.time()
t_ms = (t_stop - t_start) * 1000
print('u: ', u.shape)
print('Time: ', t_ms, ' milliseconds')

heading('Non-vectorized multiplication')
u = np.zeros((m, 1))
print('u: ', u.shape)
t_start = time.time()
for i in range(m):
    for j in range(n):
        u[i] += A[i][j] * v[j]
t_stop = time.time()
t_ms = (t_stop - t_start) * 1000
print('u: ', u.shape)
print('Time: ', t_ms, ' milliseconds')
