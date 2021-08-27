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
# def getPsiMatrix(psi, X): # for RBF sampler
#     k = psi(X[:,0]).shape[0]
#     m = X.shape[1]
#     matrix = np.empty((k,m))
#     for col in range(m):
#         matrix[:, col] = psi(X[:, col])[:, 0]
#     return matrix

# Psi_X = getPsiMatrix(psi, X_train)
# Psi_Y = getPsiMatrix(psi, Y_train)

# Psi_X_0 = getPsiMatrix(psi, X_0_train).T
# Psi_Y_0 = getPsiMatrix(psi, Y_0_train).T
# Psi_X_1 = getPsiMatrix(psi, X_1_train).T
# Psi_Y_1 = getPsiMatrix(psi, Y_1_train).T

#%% Koopman
# || Y         - X B           ||
# || Y.T       - B.T X.T       ||
# || Psi_Y_0   - K Psi_X_0     ||
# || Psi_Y_0.T - Psi_X_0.T K.T ||
K_0 = estimate_L.rrr(Psi_X_0.T, Psi_Y_0.T).T
K_1 = estimate_L.rrr(Psi_X_1.T, Psi_Y_1.T).T
eigenvalues_0, eigenvectors_0 = np.linalg.eig(K_0)
eigenvalues_1, eigenvectors_1 = np.linalg.eig(K_1)
eigenfunction_0 = list(map(lambda psi_x: np.dot(psi_x,eigenvectors_0[:,0]), Psi_X_0.T))
eigenfunction_1 = list(map(lambda psi_x: np.dot(psi_x,eigenvectors_1[:,0]), Psi_X_1.T))


plt.plot(eigenvectors_0[:,:3])
plt.plot(eigenvectors_1[:,:3])
plt.title("Eigenvectors of Koopman operator for action 0 and 1")
plt.ylabel('Eigenvector Output')
plt.xlabel('State Snapshots')
plt.show()

plt.plot(eigenfunction_0)
plt.title("Eigenfunction 0 of Koopman operator for action 0")
plt.ylabel('Eigenfunction Output')
plt.xlabel('State Snapshots')
plt.show()

plt.plot(eigenfunction_1)
plt.title("Eigenfunction 0 of Koopman operator for action 1")
plt.ylabel('Eigenfunction Output')
plt.xlabel('State Snapshots')
B = estimate_L.rrr(Psi_X.T, X_train.T, state_dim) # SINDy taking too long

#%% Prediction compounding error
title = "Prediction compounding error:"
print(title)

env = gym.make('CartPole-v0')
horizon = 1000
num_trials = 1#000
norms = []
vector_field_arrays = []
for i in range(num_trials):
    action_path = [np.random.choice([0,1]) for i in range(horizon)]
    trial_norms = []
    true_state = env.reset()
    predicted_state = true_state.copy()
    vector_field_array = [true_state]
    for h in range(horizon):
        action = action_path[h]
        psi_x = psi(predicted_state)
        predicted_state = B.T @ K_0 @ psi_x if action == 0 else B.T @ K_1 @ psi_x
        true_state, ___, __, _ = env.step(action)
        vector_field_array.append(predicted_state.reshape(state_dim))
        norm = l2_norm(true_state.reshape(-1,1), predicted_state)/l2_norm(true_state.reshape(-1,1), 0)
        trial_norms.append(norm)
    vector_field_arrays.append(vector_field_array)
    norms.append(trial_norms)
    # Test: try to populate a list with (3,) arrays and then check [:,:,1:]

vector_field_arrays = np.array(vector_field_arrays)
X_plot = vector_field_arrays[:,:,0].reshape((horizon * num_trials)+1) # cart pos
Y_plot = vector_field_arrays[:,:,2].reshape((horizon * num_trials)+1) # pole angle
U_plot = vector_field_arrays[:,:,1].reshape((horizon * num_trials)+1) # cart velocity
V_plot = vector_field_arrays[:,:,3].reshape((horizon * num_trials)+1) # pole angular velocity

plt.figure()
plt.title("Vector Field of Koopman Predicted State Evolution")
Q_plot = plt.quiver(X_plot, Y_plot, U_plot, V_plot)
plt.show()

plt.plot(np.mean(norms, axis=0))
plt.title(title)
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()


"""

#%% One-step prediction error
title = "One-step prediction error:"
print()

# data_point_index = 1000
horizon = 1000
norms = []
# action_path = U[0, data_point_index:data_point_index+horizon]
action_path = U[0, :horizon]
# starting_point = int(np.around(np.random.rand() * X_train.shape[1]))
starting_point = 1700
true_state = X[:,starting_point]
for h in range(horizon):
    action = action_path[h]
    psi_x = psi(true_state)
    predicted_state = B.T @ K_0 @ psi_x if action == 0 else B.T @ K_1 @ psi_x
    true_state = X[:,starting_point+h+1]

    norm = l2_norm(true_state.reshape(-1,1), predicted_state)/l2_norm(true_state.reshape(-1,1), 0)
    norms.append(norm)

print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.title(title)
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

#%% Error for psi(x) -> x
title = "Error for psi(x) -> x:"
print(title)

# data_point_index = 1000
horizon = 1000
norms = []
# action_path = U[0, data_point_index:data_point_index+horizon]
action_path = U[0, :horizon]
# starting_point = int(np.around(np.random.rand() * X_train.shape[1]))
starting_point = -1000
true_states = X[:,starting_point:]
for true_state in true_states.T:
    true_state = true_state.reshape(-1,1)
    projected_state = B.T @ psi(true_state)

    norm = l2_norm(true_state, projected_state)/l2_norm(true_state, 0)
    norms.append(norm)

print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.title(title)
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

#%% Error for psi(x) -> x'
title = "Error for psi(x) -> x':"
print(title)

# Koopman from psi(x) -> x'
# || Y     - X B           ||
# || Y_i   - K Psi_X_i     ||
# || Y_i.T - Psi_X_i.T K.T ||
K_0 = estimate_L.rrr(Psi_X_0.T, Y_0_train.T).T
K_1 = estimate_L.rrr(Psi_X_1.T, Y_1_train.T).T

horizon = 1000
action_path = U[0, -horizon:]
norms = []
true_states = X[:, -horizon:]
for h in range(horizon):
    action = action_path[h]
    true_state = true_states[:,h].reshape(-1,1)
    predicted_state = K_0 @ psi(true_state) if action == 0 else K_1 @ psi(true_state)

    norm = l2_norm(true_state, predicted_state)/l2_norm(true_state, 0)
    norms.append(norm)

print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.title("Error for psi(x) -> x':")
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

#%% Residual error
title = "Residual error:"
print(title)

# residuals_0 = Psi_Y_0 - Psi_X_0
# residuals_1 = Psi_Y_1 - Psi_X_1
residuals = Psi_Y - Psi_X
# psi(x) -> psi(x') - psi(x)
# K_0 = estimate_L.rrr(Psi_X_0.T, residuals_0.T).T
# K_1 = estimate_L.rrr(Psi_X_1.T, residuals_1.T).T
K = estimate_L.rrr(Psi_X.T, residuals.T).T

horizon = 1000
action_path = U[0, -horizon:]
norms = []
true_states = X_train[:, -horizon:]
true_states_prime = Y_train[:, -horizon:]
for h in range(horizon):
    action = action_path[h]

    true_state = true_states[:,h].reshape(-1,1)
    psi_x = psi(true_state)

    predicted_residual = K @ psi_x
    predicted_psi_x_prime = psi_x + predicted_residual
    predicted_x_prime = B.T @ predicted_psi_x_prime

    true_x_prime = true_states_prime[:,h].reshape(-1,1)

    norm = l2_norm(true_x_prime, predicted_x_prime)/l2_norm(true_x_prime, 0)
    norms.append(norm)

print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.title(title)
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

# Koopman from psi(x) -> psi(x')
# || Y     - X B               ||
# || Psi_Y_i   - K Psi_X_i     ||
# || Psi_Y_i.T - Psi_X_i.T K.T ||
K_0 = estimate_L.rrr(Psi_X_0.T, Psi_Y_0.T).T
K_1 = estimate_L.rrr(Psi_X_1.T, Psi_Y_1.T).T

horizon = 1000
action_path = U[0, -horizon:]
norms = []
true_states = X[:, -horizon:]
for h in range(horizon):
    action = action_path[h]
    true_state = true_states[:,h].reshape(-1,1)
    predicted_state = K_0 @ psi(true_state) if action == 0 else K_1 @ psi(true_state)

    norm = l2_norm(psi(true_state), predicted_state)/l2_norm(psi(true_state), 0)
    norms.append(norm)

print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.title("Error for psi(x) -> psi(x)':")
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

"""