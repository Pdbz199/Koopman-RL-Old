#%% Imports
import importlib
import gym
import estimate_L
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy as sp
import auxiliaries

def l2_norm(true_state, predicted_state):
    return np.sum(np.power(( true_state - predicted_state ), 2 ))

#%% Load data

X = np.load('random-agent/cartpole-states.npy').T
Y = np.append(np.roll(X, -1, axis=1)[:,:-1], np.zeros((X.shape[0],1)), axis=1)
U = np.load('random-agent/cartpole-actions.npy').reshape(1,-1)

X_0 = np.load('random-agent/cartpole-states-0.npy').T
Y_0 = np.load('random-agent/cartpole-next-states-0.npy').T
X_1 = np.load('random-agent/cartpole-states-1.npy').T
Y_1 = np.load('random-agent/cartpole-next-states-1.npy').T

num_unique_actions = 2

#%% Matrix builder functions
def phi(x):
    return x

def psi(u):
    psi_u = np.zeros((num_unique_actions))
    psi_u[u] = 1
    return psi_u

def getPhiMatrix(X):
    Phi_X = []
    for x in X[:, 100:102].T:
        Phi_X.append(phi(x))

    return np.array(Phi_X).T

def getPsiMatrix(U):
    Psi_U = []
    for u in U[:, 100:102].T:
        Psi_U.append(psi(u))

    return np.array(Psi_U).T

#%%
Phi_X = getPhiMatrix(X)
Psi_U = getPsiMatrix(U)
print(Phi_X)
print(Psi_U)

num_lifted_state_observations = Phi_X.shape[1]
num_lifted_state_features = Phi_X.shape[0]
num_lifted_action_observations = Psi_U.shape[1]
num_lifted_action_features = Psi_U.shape[0]

# @nb.njit(fastmath=True)
def getPsiPhiMatrix(Psi_U, Phi_X):
    psiPhiMatrix = np.empty((num_lifted_action_features * num_lifted_state_features, num_lifted_state_observations))

    for i in range(num_lifted_state_observations):
        kron = np.kron(Psi_U[:,i], Phi_X[:,i])
        psiPhiMatrix[:,i] = kron

    return psiPhiMatrix

psiPhiMatrix = getPsiPhiMatrix(Psi_U, Phi_X)
print("PsiPhiMatrix shape:", psiPhiMatrix.shape)
# || Y         - X B           ||
# || Phi_Y     - M PsiPhi      ||
# || Y.T       - B.T X.T       ||
# || Phi_Y.T   - PsiPhi.T M.T  ||
M = estimate_L.rrr(psiPhiMatrix.T, getPhiMatrix(Y).T).T
print("M shape:", M.shape)
assert M.shape == (num_lifted_state_features, num_lifted_state_features * num_lifted_action_features)

K = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M.shape[0]):
    K[i] = M[i].reshape((num_lifted_state_features, num_lifted_action_features))
print(K.shape)

K_u_100 = np.einsum('ijz,z->ij', K, psi(U[:,100]))
assert K_u_100.shape == (num_lifted_state_features, num_lifted_state_features)

K_u_alt = K[:,:,0]
assert np.array_equal(K_u_100, K_u_alt)

# try with quadratic feature
# LQR example