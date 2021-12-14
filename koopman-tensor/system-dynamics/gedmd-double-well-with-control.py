#%% Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
import domain
import estimate_L
import observables
import utilities

#%% Define domains
x_bounds = np.array([[-2, 2], [-1.5, 1.5]])
x_boxes = np.array([20, 15])
x_omega = domain.discretization(x_bounds, x_boxes)

u_bounds = np.array([[-2, 2]])
u_boxes = np.array([300])
u_omega = domain.discretization(u_bounds, u_boxes)

#%% Define system
def b(x):
    return np.vstack((-4*x[0, :]**3 + 4*x[0, :], -2*x[1, :]))
 
def sigma(x):
    n = x.shape[1]
    y = np.zeros((2, 2, n))
    y[0, 0, :] = 0.7
    y[0, 1, :] = x[0, :]
    y[1, 1, :] = 0.5
    return y

#%% Define observables
order = 10
phi = observables.monomials(order)
psi = observables.monomials(2)

#%% Generate data
rand_points = 100
X = x_omega.randPerBox(rand_points)
Y = b(X)
Z = sigma(X)

U = u_omega.randPerBox(rand_points)

#%% Apply dictionaries
Phi_X = phi(X)
Phi_Y = phi(Y)
Psi_U = psi(U)

dim_phi = Phi_X.shape[0]
dim_psi = Psi_U.shape[0]
N = Phi_X.shape[1]

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
n = Phi_X.shape[0] # number of basis functions
ddPhiX = phi.ddiff(X) # second-order derivatives
S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
for i in range(n):
    dPhi_Y[i, :] += 0.5*np.sum( ddPhiX[i, :, :, :] * S, axis=(0,1) )
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T
# B = estimate_L.ols(dPhi_Y.T, X.T)

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,zk->ij', K, psi(u))

#%% Plot eigenfunctions
evs = 3
_, _V = utilities.sortEig(K_u(K, np.array([[0]])).T, evs, which='SM')

fig = plt.figure()
c = x_omega.midpointGrid()
_X = c[0, :].reshape(x_omega._boxes)
_Y = c[1, :].reshape(x_omega._boxes)
Phi_c = phi(c)

for i in range(evs):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    _Z = np.real( _V[:, i].T @ Phi_c ).reshape(x_omega._boxes)
    ax.plot_surface(_X, _Y, _Z, cmap=matplotlib.cm.coolwarm)

#%%