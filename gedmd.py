#%%
from base import *

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)

# A = X' pinv(X)
# M = dPsi_X pinv(Psi_X)

#%% Step 1
U, Sigma, VT = sp.linalg.svd(Psi_X, full_matrices=False)
r = 10 # arbitrarily selected cutoff
U_tilde = U[:, :r]
Sigma_tilde = np.diag(Sigma[:r])
VT_tilde = VT[:r]

#%% Step 2
# M = dPsi_Z @ np.conj(VT_tilde).T @ sp.linalg.inv(np.diag(Sigma_tilde)) @ np.conj(U_tilde).T
# M_tilde = np.conj(U_tilde).T @ dPsi_X @ np.conj(VT_tilde).T @ sp.linalg.inv(Sigma_tilde)
M_tilde = sp.linalg.solve(Sigma_tilde.T, (U_tilde.T @ dPsi_X @ VT_tilde.T).T).T
L_tilde = M_tilde.T # estimate of Koopman generator
# print(L.shape)

#%% Eigen decomposition
eig_vals, eig_vecs = sp.sparse.linalg.eigs(L_tilde) if sp.sparse.issparse(L_tilde) else sp.linalg.eig(L_tilde)
# Compute eigenfunction matrix
# eig_funcs = (eig_vecs).T @ Psi_X

#%% Step 3
# A = M
Lambda, W = sp.linalg.eig(M_tilde) # returns W and left eigenvectors or right eigenvectors, default is right
Lambda = np.diag(Lambda)

#%% Step 4
V_v1 = X @ sp.linalg.solve(Sigma_tilde.T, VT_tilde).T @ W

# How in the world do you get eigenfunctions???
eig_funcs = []
for i in range(r):
    alpha = Sigma_tilde @ VT_tilde[:, i]
    b = sp.linalg.solve(W @ Lambda, alpha)
    eig_funcs.append((eig_vecs).T @ b)
eig_funcs = np.array(eig_funcs)

#%%
def b_v2(l, num_dims=r, V=V_v1):
    res = 0
    for ell in range(r):
        res += eig_vals[ell] * eig_funcs[ell, l] * V[:, ell] #.reshape(-1, 1)
    return np.real(res)

#%%
for l in range(r):
    b_v2_l = b_v2(l)
    print(f"b_v2({r}):", b_v2_l)
# %%