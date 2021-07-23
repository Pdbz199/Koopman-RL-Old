#%% Imports
import gym
import estimate_L
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
import scipy as sp
import auxiliaryFns


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

#%% Median trick
# num_pairs = 1000
# pairwise_distances = []
# for _ in range(num_pairs):
#     i, j = np.random.choice(np.arange(X.shape[1]), 2)
#     x_i = X[:,i]
#     x_j = X[:,j]
#     pairwise_distances.append(np.linalg.norm(x_i - x_j))
# pairwise_distances = np.array(pairwise_distances)
# gamma = np.quantile(pairwise_distances, 0.9)

# num_pairs = 1000
# pairwise_distances_u = []
# for _ in range(num_pairs):
#     i, j = np.random.choice(np.arange(U.shape[1]), 2)
#     u_i = U[0,i]
#     u_j = U[0,j]
#     pairwise_distances_u.append(np.linalg.norm(u_i - u_j))
# pairwise_distances_u = np.array(pairwise_distances_u)
# gamma_u = np.quantile(pairwise_distances_u, 0.9)

#%% RBF Sampler
# rbf_feature_x = RBFSampler(gamma=gamma, random_state=1)
# X_features = rbf_feature_x.fit_transform(X)

# def phi(x):
#     return X_features.T @ x.reshape((state_dim,1))

# rbf_feature_u = RBFSampler(gamma=gamma_u, random_state=1)
# U_features = rbf_feature_u.fit_transform(U)

# def psi(u):
#     return U_features.T @ u.reshape((1,1))


#%% 
def getPsiPhiMatrix(X,U):
    return np.kron(U,X)

M = estimate_L.rrr(Y.T, getPsiPhiMatrix(X,U).T).T

K = np.empty((state_dim,state_dim,2))
for i in range(M.shape[0]):
    K[i] = M[i].reshape(state_dim,state_dim)
     
K_0 = K[:,:,0]
K_1 = K[:,:,1]