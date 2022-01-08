#%% Imports
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../../')
import estimate_L
import observables

from control import lqr

#%% System dynamics
A = np.array([
    [0, 1],
    [-2, 2]
])
# u, s, vh = np.linalg.svd(A)
# print(s) # [2.92080963 0.68474165]
B = np.array([
    [0],
    [1]
])
# u, s, vh = np.linalg.svd(B)
# print(s) # [1.]
Q = np.array([
    [2, 0],
    [0, 2]
])
# u, s, vh = np.linalg.svd(Q)
# print(s) # [2. 2.]
R = 4

# Singular value of matrix is <1 ???

def f(x, u):
    return A @ x + B @ u

#%% Traditional LQR
lq = lqr(A, B, Q, R)
K = lq[0][0]
# lq[0] == [[ 8.19803903 12.97251466]]
# lq[1] == [[1.15898133 0.08198039]
#           [0.08198039 0.12972515]]
# lq[2] == [-9.947309 +0.j -1.0252059+0.j]

#%%
