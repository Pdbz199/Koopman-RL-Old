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

#%%
# every time you get value function and you want to do optimization
# project value function evaulated at some points (randomly sampled)
# goal is to get B as in equation 6
# same points you evaluate V, you evaluate Phi

def learningAlgorithm(L, Psi_X, reward, epsilon=0.1):
    eigenvalues, eigenvectors = sp.linalg.eig(L)
    eigenfunctions = np.dot(eigenvectors, Psi_X) # is this right?
    B = 0
    generatorModes = B.T @ sp.linalg.inv(eigenvectors).T

    summation = 0
    for ell in range(cutoff):
        summation += generatorModes[ell] * eigenvalues[ell] * eigenfunctions[ell]
    numerator = np.exp((1/lamb) * (reward + summation))
    denominator = integral_U (numerator * du)
    pi_star = numerator / denominator

    # j = 1
    V = 0
    lastV = 0
    pi_star_j = 0 # placeholder
    while abs(V - lastV) >= epsilon:
        lastV = V
        # calculate new V by pluging in last pi into equation 9