#%%
import numpy as np
from scipy import integrate
from algorithms import learningAlgorithm

#%%
dictionary_functions = [lambda x: x[0], lambda x: x[1], lambda x: x[0]**2]
def psi(X):
    output = []
    for x in X:
        row = []
        for dictionary in dictionary_functions:
            row.append(dictionary(x))
        output.append(row)
    return np.array(output).T

psi([[1,2]])

#%%
mu = -0.1
lamb = 1
L = np.array([
    [mu, 0, 0],
    [0, lamb, -lamb],
    [0, 0, 2*mu]
])
Q = np.identity(3)
R = 1

#%%
def reward(x, u):
    return -integrate.romberg(
        lambda tau, x, u: x.T(tau) @ Q @ x(tau) + u(tau).T * R * u(tau),
        0, np.inf, args=(x,u)
    )

# %%
_, pi = learningAlgorithm(
    L, X, psi, Psi_X_tilde, U, reward, timesteps=4
)