# XXX: "The @ operator can be used as a shorthand for np.matmul on ndarrays."
# XXX: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul

import numpy as np
import algorithms
from sklearn.metrics.pairwise import polynomial_kernel
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as sparse_linalg

# k = algorithms.polynomialKernel(2)
# reg_param = 0.1
# num_eigenfuncs = 4
degree = 2
gamma = 1

M = 100
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
# https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel
G_hat = polynomial_kernel(X.T, degree=degree, gamma=gamma)
A_hat = polynomial_kernel(Y.T, X.T, degree=degree, gamma=gamma)
# print(G_hat[0,0])
# print(sk_G_hat[0,0])
# test = np.array([[1,2,3,4]])
# print(polynomial_kernel(test, test, degree=2, gamma=1))

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
# Sigma_r = Sigma
Sigma_r = np.power(Sigma, 2) @ Sigma_pseudo_inverse
Sigma_r_pinv = sp.linalg.pinv(Sigma_r, rcond=1e-15)
print(f"Sigma_r shape: {Sigma_r.shape}")
Phi_X = Q @ Sigma_r @ V_hat
print(f"Phi_X shape: {Phi_X.shape}")
Xi = sp.linalg.pinv(Phi_X, rcond=1e-15) @ X.T
print(f"Xi shape: {Xi.shape}")

def varphi(x):
    varphis = []
    applied_kernels = []
    for i in range(M):
        kernel_value = polynomial_kernel([x], [X[:,i]], degree=degree, gamma=gamma)[0,0]
        applied_kernels.append(kernel_value)
    Q_times_Sigma_r_pseudo_inverse = Q @ Sigma_r_pinv
    print(f"Q @ Sigma_r_pinv shape: {Q_times_Sigma_r_pseudo_inverse.shape}")
    applied_kernels = np.reshape(applied_kernels, (1,len(applied_kernels)))
    print(f"Applied kernels shape: {applied_kernels.shape}")
    # (1000,200) times (200,) should be (1000,1) or (1000,) which it is!
    # it is (1000,)
    print(f"Q @ Sigma_r_pinv @ V_hat[0] shape: {(Q_times_Sigma_r_pseudo_inverse @ V_hat[0]).shape}")
    # (1,1000) times (1000,) should be (1,1) or (1,) which it is!
    # it is (1,)
    print((applied_kernels @ (Q_times_Sigma_r_pseudo_inverse @ V_hat[0])))
    for v_hat_k in V_hat:
        varphis.append((applied_kernels @ (Q_times_Sigma_r_pseudo_inverse @ v_hat_k))[0])
    return np.array(varphis)

# Function reconstruction
def F(x):
    varphi_calcs = varphi(x)
    print(f"varphi_calc shape: {varphi_calcs.shape}")
    summation = 0
    print("eigenvals_r[0]:", np.real(eigenvals_r[0]))
    print("Xi[0]:", Xi[0])
    print("varphi_calc[0]:", np.real(varphi_calcs[0]))
    print("eigenvals_r[0] * Xi[0]:", eigenvals_r[0] * Xi[0])
    for l in range(len(eigenvals_r)):
        summation += eigenvals_r[l] * Xi[l] * varphi_calcs[l]
    return summation

print(F([1.,2.,3.,4.])) # Expected to output [2.,3.,4.,5.] or [[2.],[3.],[4.],[5.]]



# d, V = algorithms.kedmd(X, Y, k, regularization=reg_param, evs=num_eigenfuncs)
# print(d)
# print(V)