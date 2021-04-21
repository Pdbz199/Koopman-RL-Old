#%%
# from base import np, sp, d, m, k, Psi_X, Psi_X_T, vectorized_dpsi

#%%
# Construct \text{d}\Psi_X matrix
# dPsi_X = np.empty((k, m))
# for column in range(m-1):
#     dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)
# dPsi_X_T = dPsi_X.T

#%%
import numpy as np
import scipy as sp
import numba as nb

#%% (X=Psi_X, Y=dPsi_X, rank=8)
def gedmd(X, Y, rank=8):
    U, Sigma, VT = sp.linalg.svd(X, full_matrices=False)
    U_tilde = U[:, :rank]
    Sigma_tilde = np.diag(Sigma[:rank])
    VT_tilde = VT[:rank]

    M_tilde = sp.linalg.solve(Sigma_tilde.T, (U_tilde.T @ Y @ VT_tilde.T).T).T
    L = M_tilde.T # estimate of Koopman generator
    return L

#%% (Theta=Psi_X_T, dXdt=dPsi_X_T, lamb=0.05, n=d)
def SINDy(Theta, dXdt, d, lamb=0.05):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(d):                       # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
            
    L = Xi
    return L

#%%
@nb.njit(fastmath=True)
def ols(X, Y, pinv=True):
    if pinv:
        return np.linalg.pinv(X.T @ X) @ X.T @ Y
    return np.linalg.inv(X.T @ X) @ X.T @ Y

#%% (X=Psi_X_T, Y=dPsi_X_T, rank=8)
@nb.njit(fastmath=True)
def rrr(X, Y, rank=8):
    B_ols = ols(X, Y) # if infeasible use GD (numpy CG)
    U, S, V = np.linalg.svd(Y.T @ X @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

# %%
@nb.njit(fastmath=True)
def ridgeRegression(X, y, lamb=0.05):
    return np.linalg.inv(X.T @ X + (lamb * np.identity(X.shape[1]))) @ X.T @ y