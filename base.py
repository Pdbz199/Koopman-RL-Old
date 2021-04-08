#%% Import statements
import os
import timeit
import observables
import numpy as np
import scipy as sp
import pandas as pd
from brownian import brownian

#%%
'''======================= HELPER FUNCTIONS ======================='''
# Construct B matrix as seen in 3.1.2 of the reference paper
def constructB(d, k):
    Bt = np.zeros((d, k))
    if k == 1:
        Bt[0,0] = 1
    else:
        num = np.arange(d)
        Bt[num, num+1] = 1
    B = Bt.T
    return B

# Construct similar B matrix as above, but for second order monomials
def constructSecondOrderB(s, k):
    Bt = np.zeros((s, k))
    if k == 1:
        Bt[0,0] = 1
    else:
        row = 0
        for i in range(d+1, d+1+s):
            Bt[row,i] = 1
            row += 1
    B = Bt.T
    return B

#%% Create data matrices
# The Wiener process parameter.
sigma = 1
# Total time.
T = 10000
# Number of steps.
N = 10000
# Time step size
dt = T/N
# Number of realizations to generate.
n = 20
# Create an empty array to store the realizations.
X = np.empty((n, N+1))
# Initial values of x.
X[:, 0] = 50
brownian(X[:, 0], N, dt, sigma, out=X[:, 1:])
Z = np.roll(X,-1)[:, :-1]
X = X[:, :-1]

# X is data matrix
# Z is time-delayed data matrix

#%%
d = X.shape[0]
m = X.shape[1]
s = int(d*(d+1)/2) # number of second order poly terms
rtoler=1e-02
atoler=1e-02
psi = observables.monomials(2)
Psi_X = psi(X)
Psi_X_T = Psi_X.T
k = Psi_X.shape[0]
nablaPsi = psi.diff(X)
nabla2Psi = psi.ddiff(X)
B = constructB(d, k)
second_order_B = constructSecondOrderB(s, k)

# This computes dpsi_k(x) exactly as in the paper
# t = 1 is a placeholder time step, not really sure what it should be
def dpsi(k, l, t=1):
    difference = (X[:, l+1] - X[:, l])
    term_1 = (1/t) * (difference)
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * (difference.reshape(-1, 1) @ difference.reshape(1, -1))
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1, term_2) + np.tensordot(term_3, term_4)
vectorized_dpsi = np.vectorize(dpsi)

# %%
def vectorToMatrix(vector):
    size = np.array(vector).shape[0]
    d = 1
    while size > (d*(d+1))/2:
        d += 1
    matrix = np.zeros((d, d))
    matrix[0, 0] = vector[0]
    row = 0
    col = 1
    n = 1
    while col < d and n != size:
        matrix[row, col] = vector[n]
        matrix[col, row] = vector[n]
        if row == col: 
            col += 1
            row = 0
        else:
            row += 1
        n +=1
    return matrix
# %%
