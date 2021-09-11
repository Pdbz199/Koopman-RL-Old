#%%
import numpy as np
import sys
sys.path.append('../../')
import estimate_L
import domain
import observables

#%% Define domain
bounds = np.array([[-2, 2], [-1.5, 1.5]])
boxes = np.array([20, 15])
Omega = domain.discretization(bounds, boxes)

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
psi = observables.monomials(order)

#%% Generate data
X = Omega.randPerBox(100)
Y = b(X)
Z = sigma(X)

#%% Build Phi and Psi matrices
d_phi = phi(X[:,0]).shape[0]
d_psi = phi(U[:,0]).shape[0]

Phi_X = np.empty((d_phi,N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(int(x[0]))[:,0]

Phi_Y = np.empty((d_phi,N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(int(y[0]))[:,0]

Psi_U = np.empty((d_psi,N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(int(u[0]))[:,0]

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
M =  estimate_L.ols(kronMatrix.T, dPhi_Y.T).T








#%% Apply generator EDMD
# evs = 3 # number of eigenvalues/eigenfunctions to be computed
# K, d, V = algorithms.gedmd(X, Y, Z, psi, evs=evs, operator='K')
# printVector(np.real(d), 'd')

#%% Plot eigenfunctions
# c = Omega.midpointGrid()
# Psi_c = psi(c)
# for i in range(evs):
#     plt.figure(i+1);
#     plt.clf()
#     Omega.plot(np.real( V[:, i].T @ Psi_c ), mode='3D')