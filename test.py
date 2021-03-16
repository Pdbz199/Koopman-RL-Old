import numpy as np
import algorithms
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as sparse_linalg
from scipy.spatial import distance

k = algorithms.polynomialKernel(2)
reg_param = 0.1
num_eigenfuncs = 4

M = 1000
inputs = np.zeros((4, M))
inputs[:,0] = [1,2,3,4]
for val in range(1, M):
    inputs[:,val] = inputs[:,val-1] + 1
outputs = np.roll(inputs, -1)
outputs[:,-1] = inputs[:,-1] + 1
# print(inputs)
# print(outputs)

# inputs = np.load('state-action-inputs.npy')
# outputs = np.load('state-action-outputs.npy')



# Estimate G and A matrices
G_hat = algorithms.gramian(inputs, k)
A_hat = algorithms.gramian2(inputs, outputs, k).T

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

evsK = 10
eigenvalues_K_hat, V_hat = sparse_linalg.eigs(K_hat, evsK, which='LM')
sorted_indices = eigenvalues_K_hat.argsort()[::-1]
V_hat = V_hat[:, sorted_indices]
print("\hat{V} shape:", V_hat.shape)

# Calculate matix of eigenfunction values
r = 10
eigenvals_r = np.sqrt(eigenvalues_G_hat[0:r])
zero_dummy = np.zeros((evs-r,1))
Sigma_r = np.diag(np.append(eigenvals_r, zero_dummy))
Sigma_r_pinv = sp.linalg.pinv(Sigma_r, rcond=1e-15)
print(f"Sigma_r shape: {Sigma_r.shape}")
Phi_X = Q @ Sigma_r @ V_hat
print(f"Phi_X shape: {Phi_X.shape}")
Xi = sp.linalg.pinv(Phi_X, rcond=1e-15) @ inputs.T # TODO: Ensure Xi is correct
print(f"Xi shape: {Xi.shape}")

# CORRECT UP TO THIS POINT

def varphi(x):
    varphis = []
    applied_kernels = []
    for i in range(M):
        kernel_value = k(x, inputs[:,i])
        applied_kernels.append(kernel_value)
    Q_times_Sigma_r_pseudo_inverse = Q @ Sigma_r_pinv
    print(f"Q @ Sigma_r_pinv shape: {Q_times_Sigma_r_pseudo_inverse.shape}")
    for v_hat_k in V_hat.T:
        varphis.append(np.reshape(applied_kernels, (1,len(applied_kernels))) @ (Q_times_Sigma_r_pseudo_inverse @ v_hat_k))
    return np.array(varphis)

# Function reconstruction
def F(x):
    varphi_calc = varphi(np.reshape(x, (len(x),1)))
    print(f"varphi_calc shape: {varphi_calc.shape}")
    summation = 0
    for l in range(len(eigenvals_r)):
        summation += eigenvals_r[l] * Xi[l] * varphi_calc[l]
    return summation

print(F([1,2,3,4]))

# d, V = algorithms.kedmd(inputs, outputs, k, regularization=reg_param, evs=num_eigenfuncs)
# print(d)
# print(V)