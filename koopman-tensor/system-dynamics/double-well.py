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

# make 1-dimensional system

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
psi = observables.monomials(1)

#%% Generate data
X = Omega.randPerBox(100)
Y = b(X)
Z = sigma(X)

#%% Build Phi and Psi matrices
Phi_X = phi(X)
d_phi = Phi_X[:,0].shape[0]
# d_psi = phi(U[:,0]).shape[0]
d_psi = 1
N = X.shape[1]

# Phi_X = np.empty((d_phi,N))
# for i,x in enumerate(X.T):
#     Phi_X[:,i] = phi(int(x[0]))[:,0]

# Phi_Y = np.empty((d_phi,N))
# for i,y in enumerate(Y.T):
#     Phi_Y[:,i] = phi(int(y[0]))[:,0]

dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
if not (Z is None): # stochastic dynamical system
    n = Phi_X.shape[0] # number of basis functions
    ddPhi_X = phi.ddiff(X) # second-order derivatives
    S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
    for i in range(n):
        dPhi_Y[i, :] += 0.5*np.sum( ddPhi_X[i, :, :, :] * S, axis=(0,1) )

# Psi_U = np.empty((d_psi,N))
# for i,u in enumerate(U.T):
#     Psi_U[:,i] = psi(int(u[0]))[:,0]

Psi_U = np.ones((d_psi,N))

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, [u])

#%% Training error (Mean norm on training data: 9053.466240898848)
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

norms = []
for i in range(N):
    true_phi_x_prime = dPhi_Y[:,i]
    predicted_phi_x_prime = K_u(K, 1) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())





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