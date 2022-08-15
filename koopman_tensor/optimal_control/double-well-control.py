#%%
import numpy as np
np.random.seed(123)
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

import sys
sys.path.append('../../')
import algorithmsv2
import domain
import estimate_L
import observables

def sortEig(A, evs=5):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    d, V = np.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind[:evs]], V[:, ind[:evs]])

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
# I think X and U need to be generated in a different way
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
order = 6
phi = observables.monomials(order)
psi = observables.monomials(order) #lambda u: np.array([1])

#%% Build Phi and Psi matrices
N = X.shape[1]
Phi_X = phi(X)
Psi_U = psi(U) #np.ones((1,N))
dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]

dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
ddPhi_X = phi.ddiff(X) # second-order derivatives
S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
for i in range(dim_phi):
    dPhi_Y[i, :] += 0.5*np.sum( ddPhi_X[i,:,:,:] * S, axis=(0,1) )

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    psi_u = psi(u.reshape(-1,1))[:,0]
    return np.einsum('ijz,z->ij', K, psi_u)

#%% Get eigenvalues/vectors
evs = 3
w, V = sortEig(K_u(K, np.array([0])).T, evs)

#%%
c = Omega.midpointGrid()
Phi_c = phi(c)
for i in range(evs):
    plt.figure(i+1)
    plt.clf()
    Omega.plot(np.real( V[:, i].T @ Phi_c ), mode='3D')

#%% Training error (training error decreases with lower order of monomials)
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

norms = []
for i in range(N):
    true_phi_x_prime = dPhi_Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())

#%% Define cost function
def cost(x, u):
    if x.shape == 2:
        return x[0,0]**2
    return x[0]**2

#%% Discretize all controls
def discretize(start, end, num_points):
    step_size = (np.abs(start) + np.abs(end)) / num_points
    ret = [start]
    for i in range(1,num_points):
        ret.append(ret[i-1] + step_size)
    return ret

U = []
for i in range(41):
    U.append([-2 + (i * 0.1)])
U = np.array(U)

#%% Control
algos = algorithmsv2.algos(X, U, u_bounds[0], u_bounds[1], phi, psi, K, cost, epsilon=1)
pi = algos.algorithm2()
# pi = algos.algorithm3()

#%% Bellman Errors
# 1184.3180405984
# 3508912268.71883