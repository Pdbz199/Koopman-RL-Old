#%%
from base import *

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)

#%%
def sparsifyDynamics(Theta, dXdt, lamb, n):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(n):                       # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
            
    return Xi

#%%
lamb = 0.05 # sparsification knob lambda
Xi = sparsifyDynamics(Psi_X.T, dPsi_X.T, lamb, d)
#%%
L = Xi # estimate of Koopman generator
L_T = L.T

eigenvalues, eigenvectors = sp.linalg.eig(L_T)


# from scipy.misc import derivative
# def f(x):
#     return x**3 + x**2
# derivative(f, 1.0, dx=1e-6)
# 4.9999999999217337
def rhoV(x):
    # suprema_{pi_t \in P(U)} integral_U r(x,u) - \
    #     lamb * np.log(pi_t(u)) * du + integral_U \sum_{l=1}^cutoff \
    #         lamb * eig_funcs[ell](x, u) * m_ell^V * pi_t(u) du
    pass