#%% Import statements
import os
import observables
import numpy as np
import scipy as sp
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
k = Psi_X.shape[0]
nablaPsi = psi.diff(X)
nabla2Psi = psi.ddiff(X)

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

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)

#%%
# A = X' pinv(X)
# M = dPsi_X pinv(Psi_X)

# Step 1
U, Sigma, VT = sp.linalg.svd(Psi_X, full_matrices=False)
r = 10 # arbitrarily selected cutoff
U_tilde = U[:, :r]
Sigma_tilde = Sigma[:r]
VT_tilde = VT[:r]

# Step 2
# M = dPsi_Z @ np.conj(VT_tilde).T @ sp.linalg.inv(np.diag(Sigma_tilde)) @ np.conj(U_tilde).T
M_tilde = np.conj(U_tilde).T @ dPsi_X @ np.conj(VT_tilde).T @ sp.linalg.inv(np.diag(Sigma_tilde))
L = M_tilde.T # estimate of Koopman generator
# print(L.shape)

#%%
# Eigen decomposition
eig_vals, eig_vecs = sp.sparse.linalg.eigs(L) if sp.sparse.issparse(L) else sp.linalg.eig(L)
# Compute eigenfunction matrix
eig_funcs = (eig_vecs).T @ Psi_X

# Construct B matrix that selects first-order monomials (except 1) when multiplied by list of dictionary functions
B = constructB(d, k)
# Construct second order B matrix (selects second-order monomials)
second_orderB = constructSecondOrderB(s, k)

# TODO: Calculate Koopman generator modes using steps 3 and 4
V_v1 = B.T @ np.linalg.inv((eig_vecs).T)
# The b_v2 function allows for heavy dimension reduction
# default is reducing by 90% (taking the first k/10 eigen-parts)
# TODO: Figure out correct place to take reals
def b_v2(l, num_dims=r, V=V_v1):
    res = 0
    for ell in range(k-1, k-num_dims, -1):
        res += eig_vals[ell] * eig_funcs[ell, l] * V[:, ell] #.reshape(-1, 1)
    return np.real(res)

#%%
for l in range(m):
    # b_l = b(l)
    # print("b(l):", b_l)
    b_v2_l = b_v2(l)
    print("b_v2(l):", b_v2_l)