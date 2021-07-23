#%% Imports
import importlib
import gym
estimate_L = importlib.import_module("estimate-L")
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
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

state_dim = X.shape[0]
d = X.shape[1]

percent_training = 0.25
train_ind = int(np.around(X.shape[1]*percent_training))
X_train = X[:,:train_ind]
Y_train = Y[:,:train_ind]
train_inds = [
    int(np.around(X_0.shape[1]*percent_training)),
    int(np.around(X_1.shape[1]*percent_training))
]
X_0_train = X_0[:,:train_inds[0]]
X_1_train = X_1[:,:train_inds[1]]
Y_0_train = Y_0[:,:train_inds[0]]
Y_1_train = Y_1[:,:train_inds[1]]

#%% Matrix builder functions
def phi(x):
    return x

def psi(u):
    psi_u = np.zeros((1,2))
    psi_u[u] = 1
    return psi_u

def getPhiMatrix(X):
    return X

def getPsiMatrix(U):
    return U

#%%
Psi_U = getPsiMatrix(U)
Phi_X = getPhiMatrix(X)

@nb.njit(fastmath=True)
def getPsiPhiMatrix(U,X):
    return np.kron(Psi_U, Phi_X)

M = estimate_L.rrr(Y.T, getPsiPhiMatrix(U,X).T).T
# assert d x d^2

# take a look into this with simple example
K = np.empty((d,d,2))
for i in range(M.shape[0]):
    K[i] = M[i].reshape(state_dim,state_dim)

K_0 = K[:,:,0]
K_1 = K[:,:,1]

# try with quadratic feature
# LQR example