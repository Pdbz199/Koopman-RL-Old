#%% Imports
import algorithms
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

#%% Load data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)
X_0 = []
Y_0 = []
X_1 = []
Y_1 = []
for i in range(U.shape[1]):
    action = U[0,i]
    if action == 0 and i != U.shape[1]-1:
        X_0.append(X[:,i])
        Y_0.append(X[:,i+1])
    elif action == 1 and i != U.shape[1]-1:
        X_1.append(X[:,i])
        Y_1.append(X[:,i+1])

X_0 = np.array(X_0).T
Y_0 = np.array(Y_0).T
X_1 = np.array(X_1).T
Y_1 = np.array(Y_1).T

X_0_train = X_0[:,:1000]
Y_0_train = Y_0[:,:1000]
X_1_train = X_1[:,:1000]
Y_1_train = Y_1[:,:1000]

#%% RBF Sampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_0_features = rbf_feature.fit_transform(X_0)
X_1_features = rbf_feature.fit_transform(X_1)
# k_0 = X_0_features.shape[1]
# k_1 = X_1_features.shape[1]
psi_0 = lambda x: X_0_features.T @ x.reshape(-1,1)
psi_1 = lambda x: X_1_features.T @ x.reshape(-1,1)

#%% Psi matrices
def getPsiMatrix(psi, X):
    k = psi(X[:,0].reshape(-1,1)).shape[0]
    m = X.shape[1]
    matrix = np.empty((k,m))
    for col in range(m):
        matrix[:, col] = psi(X[:, col])[:, 0]
    return matrix

Psi_X_0 = getPsiMatrix(psi_0, X_0_train)
Psi_Y_0 = getPsiMatrix(psi_0, Y_0_train)
Psi_X_1 = getPsiMatrix(psi_1, X_1_train)
Psi_Y_1 = getPsiMatrix(psi_1, Y_1_train)

#%% Koopman
# || Y         - X B           ||
# || Y.T       - B.T X.T       ||
# || Psi_Y_0   - K Psi_X_0     ||
# || Psi_Y_0.T - Psi_X_0.T K.T ||
K_0 = algorithms.rrr(Psi_X_0.T, Psi_Y_0.T).T
K_1 = algorithms.rrr(Psi_X_1.T, Psi_Y_1.T).T

#%% Find mapping from Psi_X to X
B_0 = algorithms.SINDy(Psi_X_0.T, X_0_train.T, X_0_train.shape[0])
B_1 = algorithms.SINDy(Psi_X_1.T, X_1_train.T, X_1_train.shape[0])

#%%
import gym
env = gym.make('CartPole-v0')
horizon = 1000
data_point_index = 2000
action_path = U[0, data_point_index:data_point_index+horizon]
norms = []
true_state = env.reset()
predicted_state = true_state.copy()
for h in range(horizon):
    action = action_path[h]
    if action == 0:
        predicted_state = B_0.T @ K_0 @ psi_0(predicted_state.reshape(-1,1))
    else:
        predicted_state = B_1.T @ K_1 @ psi_1(predicted_state.reshape(-1,1))
    true_state, ___, __, _ = env.step(action)

    predicted_state_norm = np.linalg.norm(predicted_state)
    true_state_norm = np.linalg.norm(true_state)
    norms.append(np.absolute(true_state_norm - predicted_state_norm))
norms = np.array(norms)

# %%
import matplotlib.pyplot as plt
plt.plot(norms)
plt.ylabel('Difference between norms')
plt.xlabel('Timestep')
plt.show()

# %%
