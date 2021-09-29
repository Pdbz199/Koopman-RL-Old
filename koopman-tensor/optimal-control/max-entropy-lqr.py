# https://stephentu.github.io/blog/optimal-control/2018/01/08/max-entropy-lqr.html
# Based on DMDc problem but with entropy regularization

#%% Imports
import numpy as np
import scipy.integrate as integrate
import sys
sys.path.append('../../')
import estimate_L

#%% Definitions
A = np.array([
    [1.5, 0],
    [0, 0.1]
])
B = np.array([
    [1],
    [0]
])
Q = np.identity(2)
R = 1

def F(x,u):
    return A @ x.reshape(-1, 1) + B @ u.reshape(-1, 1)

def control(x):
    """ compute state-dependent control variable u """
    u = -1*x[0] + np.random.randn()
    return u.reshape(-1, 1)

def phi(x):
    """ Quadratic dictionary """
    return np.array([1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]])

def psi(u):
    """ Quadratic dictionary """
    return np.array([float(1), float(u), float(u**2)])

#! check on the H term (integral)
u_bounds = [-50, 50]
def cost(x, u, pi):
    return 0.5 * (x.T @ Q @ x + u.T @ R @ u) + (0.5 * x.T @ Q @ x) - integrate.quad(pi, u_bounds[0], u_bounds[1], (x))

# simulate system to generate data matrices
m = 1000 # number of sample steps from the system.
n = 2 # dimensionality of state space
q = 1 # dimensionality of control space

#%% State snapshotting
x0 = np.array([
    [4],
    [7]
])
snapshots = np.empty((n, m))
snapshots[:, 0] = np.squeeze(x0)

# Control snapshotting
U = np.empty((q, m-1))
# sys = UnstableSystem1(x0)
for k in range(m-1):
    u_k = control(snapshots[:, k])
    y = F(snapshots[:, k], u_k[0])
    snapshots[:, k+1] = np.squeeze(y)
    U[:, k] = u_k

X = snapshots[:, :m-1]
Y = snapshots[:, 1:m]

#%% Build Phi and Psi matrices
d_phi = 6
d_psi = 3
N = m-1

Phi_X = np.empty((d_phi, N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(x)

Phi_Y = np.empty((d_phi, N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(y)

Psi_U = np.empty((d_psi, N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(u)

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(u))

#%% Training error
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

norms = []
for i in range(N):
    true_phi_x_prime = Phi_Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())