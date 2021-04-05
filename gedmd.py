#%%
from base import *
import matplotlib.pyplot as plt

# A = X' pinv(X)
# M = dPsi_X pinv(Psi_X)

#%% Step 1
U, Sigma, VT = sp.linalg.svd(Psi_X, full_matrices=False)
plt.plot(Sigma)
plt.axline((7, 1e7), (7, 0), color='b')
plt.show()
r = 8 # selected from elbow in graph (not sure if plotted correctly though)
U_tilde = U[:, :r]
Sigma_tilde = np.diag(Sigma[:r])
VT_tilde = VT[:r]

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)

#%% Step 2
# M = dPsi_Z @ np.conj(VT_tilde).T @ sp.linalg.inv(np.diag(Sigma_tilde)) @ np.conj(U_tilde).T
# M_tilde = np.conj(U_tilde).T @ dPsi_X @ np.conj(VT_tilde).T @ sp.linalg.inv(Sigma_tilde)
M_tilde = sp.linalg.solve(Sigma_tilde.T, (U_tilde.T @ dPsi_X @ VT_tilde.T).T).T
L_tilde = M_tilde.T # estimate of Koopman generator
# print(L.shape)

#%% Eigen decomposition
eig_vals, eig_vecs = sp.sparse.linalg.eigs(M_tilde) if sp.sparse.issparse(M_tilde) else sp.linalg.eig(M_tilde)
# Compute eigenfunction matrix
# eig_funcs = (eig_vecs).T @ Psi_X

#%% Step 3
# A = M
Lambda, W = sp.linalg.eig(M_tilde) # returns W and left eigenvectors or right eigenvectors, default is right
Lambda = np.diag(Lambda)

#%% Step 4
V_v1 = dPsi_X @ sp.linalg.solve(Sigma_tilde.T, VT_tilde).T @ W
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

#%%
V_v2 = second_order_B.T @ np.linalg.inv((eig_vecs).T)
def a_v2(l):
    return (b_v2(l, V=V_v2)) - \
        (second_order_B.T @ nablaPsi[:, :, l] @ b_v2(l))

#%% Reshape a vector as matrix and perform some tests
def covarianceMatrix(a_func, l):
    a_l = a_func(l)
    covariance = np.zeros((d, d))
    row = 0
    col = 0
    covariance[row, col] = a_l[0]
    col += 1
    n = 1
    while col < d:
        covariance[row, col] = a_l[n]
        covariance[col, row] = a_l[n]
        if row == col: 
            col += 1
            row = 0
        else:
            row += 1
        n +=1
    return covariance

test_v2 = covarianceMatrix(a_v2, 2)
test_v2_df = pd.DataFrame(test_v2)
print("a_v2:", test_v2_df)
print("a_v2 diagonal:", np.diagonal(test_v2))