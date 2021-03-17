# XXX: the @ symbol sometimes produces array values when it shouldn't
# XXX: Not really sure what the solution is...

import numpy as np
import algorithms
from sklearn.metrics.pairwise import polynomial_kernel
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as sparse_linalg

k = algorithms.polynomialKernel(2) # from one of the research repos, change to scikit?
reg_param = 0.1
num_eigenfuncs = 4

M = 1000
# random initial state?
# potential constant deviation
X = np.zeros((4, M))
X[:,0] = [1,2,3,4] # sample from distribution, add one on it for a while
for val in range(1, M):
    X[:,val] = X[:,val-1] + 1
Y = np.roll(X, -1)
Y[:,-1] = X[:,-1] + 1
# print(X)
# print(Y)

# X = np.load('state-action-inputs.npy')
# Y = np.load('state-action-outputs.npy')



# Estimate G and A matrices
# TODO: Look more closely at gramians, maybe use sklearn?
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html
G_hat = algorithms.gramian(X, k)
A_hat = algorithms.gramian2(Y, X, k)

# Get Q and \Sigma from \hat{G}
evs = 200
eigenvalues_G_hat, eigenvectors_G_hat = sparse_linalg.eigs(G_hat, evs, which='LM')
sorted_indices = eigenvalues_G_hat.argsort()[::-1]
eigenvalues_G_hat = eigenvalues_G_hat[sorted_indices]
Q = eigenvectors_G_hat[:, sorted_indices]
print(f"Q shape: {Q.shape}")
Sigma = np.diag(np.sqrt(eigenvalues_G_hat))
print(f"Sigma shape: {Sigma.shape}")

# Estimate Koopman operator
Sigma_pseudo_inverse = sp.linalg.pinv(Sigma, rcond=1e-15)
K_hat = (Sigma_pseudo_inverse @ Q.T) @ A_hat @ (Q @ Sigma_pseudo_inverse)
print("\hat{K} shape:", K_hat.shape)

# evsK = 10
eigenvalues_K_hat, V_hat = sparse_linalg.eigs(K_hat, evs, which='LM')
sorted_indices = eigenvalues_K_hat.argsort()[::-1]
V_hat = V_hat[:, sorted_indices]
print("\hat{V} shape:", V_hat.shape)

# Calculate matix of eigenfunction values
# r = 10
eigenvals_r = np.sqrt(eigenvalues_G_hat)
# zero_dummy = np.zeros((evs-r,1))
# Sigma_r = np.diag(np.append(eigenvals_r, zero_dummy))
Sigma_r = Sigma
# Sigma_r = np.power(Sigma, 2) @ Sigma_pseudo_inverse
Sigma_r_pinv = sp.linalg.pinv(Sigma_r, rcond=1e-15)
print(f"Sigma_r shape: {Sigma_r.shape}")
Phi_X = Q @ Sigma_r @ V_hat
print(f"Phi_X shape: {Phi_X.shape}")
Xi = sp.linalg.pinv(Phi_X, rcond=1e-15) @ X.T
print(f"Xi shape: {Xi.shape}")

# TODO: Fix varphi calc
def varphi(x):
    varphis = []
    applied_kernels = []
    for i in range(M):
        kernel_value = k(x, X[:,i].reshape(X[:,i].shape[0],-1))
        applied_kernels.append(kernel_value[0,0])
    Q_times_Sigma_r_pseudo_inverse = Q @ Sigma_r_pinv
    print(f"Q @ Sigma_r_pinv shape: {Q_times_Sigma_r_pseudo_inverse.shape}")
    applied_kernels = np.reshape(applied_kernels, (len(applied_kernels),-1))
    for v_hat_k in V_hat:
        varphis.append(applied_kernels @ (Q_times_Sigma_r_pseudo_inverse @ v_hat_k))
    return np.array(varphis)

# Function reconstruction
def F(x):
    # TODO: Determine whether varphi takes in column vector (as is now) or row vector
    varphi_calc = varphi(np.reshape(x, (len(x),-1)))
    print(f"varphi_calc shape: {varphi_calc.shape}")
    # TODO: Check summation calculation (supposed to be using "*"?)
    summation = 0
    for l in range(len(eigenvals_r)):
        summation += eigenvals_r[l] * Xi[l] * varphi_calc[l]
    return summation

print(F([1.,2.,3.,4.])) # Expected to output [2.,3.,4.,5.] or [[2.],[3.],[4.],[5.]]



# d, V = algorithms.kedmd(X, Y, k, regularization=reg_param, evs=num_eigenfuncs)
# print(d)
# print(V)