#%% gEDMD 2D Double Well W/ Control

#%% Imports
import numpy as np
np.random.seed(123)
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

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
def b(x, u):
    return np.vstack([-4*x[0, :]**3 + 4*x[0, :], -2*x[1, :]]) + u
 
def sigma(x):
    n = x.shape[1]
    y = np.zeros((2, 2, n))
    y[0, 0, :] = 0.7
    y[0, 1, :] = x[0, :]
    y[1, 1, :] = 0.5
    return y

#%% Define observables
order = 6
phi = observables.monomials(order)
psi = observables.monomials(order)

#%% Generate data
rand_points = 100
X = x_omega.randPerBox(rand_points)
U = u_omega.randPerBox(rand_points)
Y = b(X, U)
Z = sigma(X)



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

#%% Compute values for plotting eigenfunctions
fig = plt.figure()
c = x_omega.midpointGrid()
_X = c[0, :].reshape(x_omega._boxes)
_Y = c[1, :].reshape(x_omega._boxes)
Phi_c = phi(c)

evs = 3

axes = []
for i in range(evs):
    axes.append(fig.add_subplot(1, evs+1, i+1+1, projection='3d'))

#%%
# fig = plt.figure()
# _, _V = utilities.sortEig(K_u(K, np.array([[0]])).T, evs, which='SM')
# for i in range(evs):
#     ax = fig.add_subplot(1, 3, i+1, projection='3d')
#     _Z = np.real( _V[:, i].T @ Phi_c ).reshape(x_omega._boxes)
#     ax.plot_surface(_X, _Y, _Z, cmap=matplotlib.cm.coolwarm)

#%% Define initial parameters
axcolor = 'lightgoldenrodyellow'

init_c = 0

#%%
ax_c = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor=axcolor)
c_slider = Slider(
    ax=ax_c,
    label="c",
    valmin=-2,
    valmax=2,
    valinit=init_c,
    orientation="vertical"
)

#%%
# The function to be called anytime a slider's value changes
def update(val):
    _, _V = utilities.sortEig(K_u(K, np.array([[c_slider.val]])).T, evs, which='SM')

    for i in range(evs):
        axes[i].clear()
        _Z = np.real( _V[:, i].T @ Phi_c ).reshape(x_omega._boxes)
        axes[i].plot_surface(_X, _Y, _Z, cmap=matplotlib.cm.coolwarm)

#%% Register the update function with each slider
c_slider.on_changed(update)

#%%
while(True):
    update(0)
    plt.pause(0.01)

#%%