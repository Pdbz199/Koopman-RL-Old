import numpy as np
import algorithms
import matplotlib.pyplot as plt

k = algorithms.gaussianKernel(np.sqrt(0.3))
reg_param = 0.1
num_eigenfuncs = 4

f = np.vectorize(lambda x: x**2)

inputs = np.zeros((100, 2))
for val in range(inputs.shape[0]):
    inputs[val] = [val+1, val+2]
inputs = np.transpose(inputs)
outputs = f(inputs)

d, V = algorithms.kedmd(inputs, outputs, k, regularization=reg_param, evs=num_eigenfuncs)

# TODO: Find out what to do with evaluated eigenfunctions

# \mu is eigenvalue \xi is koopman mode \varphi is eigenfunction
# \sum_{k=1}^K \mu_k \xi_k \varphi_k (x)
def reconstructed(x):
    summation = 0
    for k in range(K):
        summation += d[k] * xi[k] * V[:, k]
    return summation