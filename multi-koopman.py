#%% Imports
import algorithms
import numpy as np

#%% Load data
# X = np.load('optimal-agent/cartpole-states.npy').T
# U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)
# X_0 = []
# Y_0 = []
# X_1 = []
# Y_1 = []
# for i in range(U.shape[1]):
#     action = U[0,i]
#     if action == 0 and i != U.shape[1]-1:
#         X_0.append(X[:,i])
#         Y_0.append(X[:,i+1])
#     elif action == 1 and i != U.shape[1]-1:
#         X_1.append(X[:,i])
#         Y_1.append(X[:,i+1])

# X_0 = np.array(X_0).T
# Y_0 = np.array(Y_0).T
# X_1 = np.array(X_1).T
# Y_1 = np.array(Y_1).T

X = np.load('random-agent/cartpole-states.npy').T
U = np.load('random-agent/cartpole-actions.npy').reshape(1,-1)

X_0 = np.load('random-agent/cartpole-states-0.npy').T
Y_0 = np.load('random-agent/cartpole-next-states-0.npy').T
X_1 = np.load('random-agent/cartpole-states-1.npy').T
Y_1 = np.load('random-agent/cartpole-next-states-1.npy').T

state_dim = X.shape[0]

percent_training = 0.8
train_ind = int(np.around(X.shape[1]*percent_training))
X_train = X[:,:train_ind]
train_inds = [
    int(np.around(X_0.shape[1]*percent_training)),
    int(np.around(X_1.shape[1]*percent_training))
]
X_0_train = X_0[:,:train_inds[0]]
X_1_train = X_1[:,:train_inds[1]]
Y_0_train = Y_0[:,:train_inds[0]]
Y_1_train = Y_1[:,:train_inds[1]]

#%% Median trick
num_pairs = 1000
pairwise_distances = []
for _ in range(num_pairs):
    i, j = np.random.choice(np.arange(X.shape[1]), 2)
    x_i = X[:,i]
    x_j = X[:,j]
    pairwise_distances.append(np.linalg.norm(x_i - x_j))
pairwise_distances = np.array(pairwise_distances)
gamma = np.quantile(pairwise_distances, 0.9)

#%% RBF Sampler
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=gamma, random_state=1)
X_features = rbf_feature.fit_transform(X)
def psi(x):
    return X_features.T @ x.reshape((state_dim,1))
# X_0_features = rbf_feature.fit_transform(X_0)
# X_1_features = rbf_feature.fit_transform(X_1)
# k_0 = X_0_features.shape[1]
# k_1 = X_1_features.shape[1]
# psi_0 = lambda x: X_features.T @ x.reshape(-1,1)
# psi_1 = lambda x: X_features.T @ x.reshape(-1,1)

#%% Nystroem
# from sklearn.kernel_approximation import Nystroem
# feature_map_nystroem = Nystroem(gamma=0.7, random_state=1, n_components=4)
# data_transformed = feature_map_nystroem.fit_transform(X)
# psi = lambda x: data_transformed @ x.reshape(-1,1)

#%% Psi matrices
def getPsiMatrix(psi, X):
    k = psi(X[:,0].reshape(-1,1)).shape[0]
    m = X.shape[1]
    matrix = np.empty((k,m))
    for col in range(m):
        matrix[:, col] = psi(X[:, col])[:, 0]
    return matrix

Psi_X_0 = getPsiMatrix(psi, X_0_train)
Psi_Y_0 = getPsiMatrix(psi, Y_0_train)
Psi_X_1 = getPsiMatrix(psi, X_1_train)
Psi_Y_1 = getPsiMatrix(psi, Y_1_train)

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

#%% Prediction compounding error
import gym
env = gym.make('CartPole-v0')
horizon = 1000
num_trials = 1#000
norms = []
for i in range(num_trials):
    action_path = [np.random.choice([0,1]) for i in range(horizon)]
    trial_norms = []
    true_state = env.reset()
    predicted_state = true_state.copy()
    for h in range(horizon):
        action = action_path[h]
        psi_x = psi(predicted_state)
        predicted_state = B_0.T @ K_0 @ psi_x if action == 0 else B_1.T @ K_1 @ psi_x
        true_state, ___, __, _ = env.step(action)

        norm = np.sum( np.power( ( true_state.reshape(-1,1) - predicted_state ) , 2 ) )
        trial_norms.append(norm)
    norms.append(trial_norms)

#%%
import matplotlib.pyplot as plt
plt.plot(np.mean(norms, axis=0))
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

#%% One-step prediction error
# data_point_index = 1000
horizon = 1000
norms = []
# action_path = U[0, data_point_index:data_point_index+horizon]
action_path = U[0, :horizon]
norms = []
# starting_point = int(np.around(np.random.rand() * X_train.shape[1]))
starting_point = 1700
true_state = X[:,starting_point]
for h in range(horizon):
    action = action_path[h]
    psi_x = psi(true_state)
    predicted_state = B_0.T @ K_0 @ psi_x if action == 0 else B_1.T @ K_1 @ psi_x
    true_state = X[:,starting_point+h+1]

    norm = np.sum( np.power( ( true_state.reshape(-1,1) - predicted_state ) , 2 ) )
    norms.append(norm)

#%%
print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

#%%
