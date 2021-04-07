#%%
import numpy as np
import scipy as sp

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
# %%
