#%%
from base import k, m, vectorized_dpsi

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)
dPsi_X_T = dPsi_X.T

#%%
def gedmd(X=Psi_X, Y=dPsi_X, rank=8):
    U, Sigma, VT = sp.linalg.svd(X, full_matrices=False)
    U_tilde = U[:, :rank]
    Sigma_tilde = np.diag(Sigma[:rank])
    VT_tilde = VT[:rank]

    M_tilde = sp.linalg.solve(Sigma_tilde.T, (U_tilde.T @ Y @ VT_tilde.T).T).T
    L = M_tilde.T # estimate of Koopman generator
    return L

#%%
def SINDy(Theta=Psi_X_T, dXdt=dPsi_X_T, lamb=0.05, n=d):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(n):                       # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
            
    L = Xi
    return L

#%%
def rrr(X=Psi_X_T, Y=dPsi_X_T, rank=8):
    B_ols = sp.linalg.inv(X.T @ X) @ X.T @ Y
    U, S, V = sp.linalg.svd(dPsi_X @ Psi_X_T @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

# %%
