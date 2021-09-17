#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys
sys.path.append('../../')
import domain
import estimate_L
import observables

class DoubleWell():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return -x**3 + 2*x + self.c
    
    def sigma(self, x):
        return np.sqrt(2/self.beta)
    
s = DoubleWell(beta=1, c=0)
h = 1e-2
y = 1

x = np.linspace(-2.5, 2.5, 1000)

# The parametrized function to be plotted
def f(x, beta, c):
    return 1/4*x**4 - x**2 - c*x

# Define initial parameters
init_beta = 1
init_c = 0
u_bounds = [-2, 2]
bounds = np.array([u_bounds, [-1.5, 1.5]])
boxes = np.array([20, 15])
Omega = domain.discretization(bounds, boxes)
X = Omega.randPerBox(100)
U = np.random.uniform(u_bounds[0], u_bounds[1], (1, X.shape[1]))

dim_x = X.shape[0]
dim_u = U.shape[0]

Y = np.empty(X.shape)
Z = np.empty((2,2,X.shape[1]))
for i in range(X.shape[1]):
    s.c = U[0, i]
    # X_prime[:, i] = f(X[:, 0], s.beta, s.c)
    # Y[:, i] = y + s.b(y)*h + s.sigma(y)*np.sqrt(h)*np.random.randn()
    Y[:, i] = s.b(X[:, i])
    Z[0, 0, i] = s.sigma(X[:, i])
    Z[0, 1, i] = 0
    Z[1, 1, i] = s.sigma(X[:, i])
    Z[1, 0, i] = 0

#%% Define observables
order = 10
phi = observables.monomials(order)
psi = observables.monomials(order)

#%% Build Phi and Psi matrices
N = X.shape[1]
Phi_X = phi(X)
Psi_U = psi(U)
dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]

dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
ddPhi_X = phi.ddiff(X) # second-order derivatives
S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
for i in range(dim_phi):
    dPhi_Y[i, :] += 0.5*np.sum( ddPhi_X[i, :, :, :] * S, axis=(0,1) )

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, u)

# realizing that the when action is 0 the K_u = [0.] which, I think, means we need a different dictionary

#%% Training error (Mean norm on training data: 556.7067749613773)
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

norms = []
for i in range(N):
    true_phi_x_prime = dPhi_Y[:,i]
    predicted_phi_x_prime = K_u(K, Psi_U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())

#%% Plotting eigenfunctions
action_0 = psi(np.array([[0]]))[:,0]
K_0 = K_u(K, action_0)

w, V = np.linalg.eig(K_0)
w = np.real(w)
V = np.real(V)
eigenfunction_i = lambda i, phi_x: V[:,i].T @ phi_x

eigenfunction_0 = eigenfunction_i(0, Phi_X)
# print(l2_norm(V, V.T)) # 113.90167782677221