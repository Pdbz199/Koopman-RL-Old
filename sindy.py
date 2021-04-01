#%%
from base import *

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)

#%%
def sparsifyDynamics(Theta,dXdt,lamb,n):
    Xi = np.linalg.lstsq(Theta,dXdt,rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(n):                       # n is state dimension
            biginds = smallinds[:,ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],dXdt[:,ind],rcond=None)[0]
            
    return Xi

#%%
lamb = 0.025 # sparsification knob lambda
Xi = sparsifyDynamics(Psi_X, dPsi_X, lamb, 3) # all 0s
# %%
