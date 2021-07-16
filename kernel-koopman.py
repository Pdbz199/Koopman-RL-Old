#%% Imports
import auxiliaries
import gym
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

#%% Import and process data
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

#%% ExpSineSquared (periodic kernel)
from sklearn.gaussian_process.kernels import ExpSineSquared
kernel = ExpSineSquared(length_scale=1, periodicity=1)

#%% Build gramians
def getGramian(X, Y):
    return kernel(X.T, Y.T)

G_0 = getGramian(X, X)
G_1 = getGramian(X, Y).T

epsilon = 0.0
thresh = 5
# A = np.linalg.pinv(G_0 + epsilon*_np.eye(n), rcond=1e-15) @ G_1
# d, V = auxiliaries.sortEig(A, thresh)

#%% According to algorithm 3 (Williams 2015)
G_hat = G_0
A_hat = G_1

d, V = auxiliaries.sortEig(G_hat, thresh) # Q Sigma^2 Q^T
Q = V
Sigma = np.sqrt(np.diag(d))

Sigma_pinv = np.linalg.pinv(Sigma)
K_hat = (Sigma_pinv @ Q.T) @ A_hat @ (Q @ Sigma_pinv)

#%% Koopman eigenfunction values
d, V_hat = auxiliaries.sortEig(K_hat)
Phi_x = Q @ Sigma @ V_hat
def eigenfunction(k, x): # evaluate eigenfunction k with state x
    if x.shape[0] != 1:
        x = x.reshape(1,x.shape[0])
    return kernel(x, X.T) @ (Q @ Sigma_pinv @ V_hat[:,k])

#%% Koopman modes
koopman_modes = np.linalg.inv(V_hat) @ Sigma_pinv @ Q.T @ X.T

#%% model of function
def model(x, threshold=thresh):
    predicted_state = np.zeros((1,koopman_modes.shape[1]))
    for k in range(threshold):
        predicted_state += np.real(d[k] * koopman_modes[k] * eigenfunction(k, x))
    return predicted_state.reshape(-1,1)

#%% One-step prediction error
title = "One-step prediction error:"
print(title)

# data_point_index = 1000
horizon = 10000
action_path = U[0, :horizon]
starting_point = 1700
true_state = X[:,starting_point]
norms = []
h = 0
while h < horizon:
    action = action_path[h]
    predicted_state = model(true_state) # column vector
    true_state = X[:,starting_point+1+h].reshape(-1,1) # column vector

    norm = auxiliaries.l2_norm(true_state, predicted_state)/auxiliaries.l2_norm(true_state, 0)
    norms.append(norm)

    if h % 100 == 20: h += 180
    else: h += 1

print("Mean norm:", np.mean(norms))
plt.plot(norms, marker='.', linestyle='')
plt.title(title)
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()

#%% Prediction compounding error
title = "Prediction compounding error:"
print(title)

env = gym.make('CartPole-v0')
horizon = 1000
num_trials = 10#00
norms = []
for i in range(num_trials):
    action_path = [np.random.choice([0,1]) for i in range(horizon)]
    trial_norms = []
    true_state = env.reset()
    predicted_state = true_state.copy()
    # vector_field_array = [true_state]
    for h in range(horizon):
        action = action_path[h]
        predicted_state = model(predicted_state)
        true_state = (env.step(action)[0]).reshape(-1,1)
        # vector_field_array.append(predicted_state.reshape(state_dim))
        norm = auxiliaries.l2_norm(true_state, predicted_state)/auxiliaries.l2_norm(true_state, 0)
        trial_norms.append(norm)
    # vector_field_arrays.append(vector_field_array)
    norms.append(trial_norms)

# vector_field_arrays = np.array(vector_field_arrays)
# X_plot = vector_field_arrays[:,:,0].reshape((horizon * num_trials)+1) # cart pos
# Y_plot = vector_field_arrays[:,:,2].reshape((horizon * num_trials)+1) # pole angle
# U_plot = vector_field_arrays[:,:,1].reshape((horizon * num_trials)+1) # cart velocity
# V_plot = vector_field_arrays[:,:,3].reshape((horizon * num_trials)+1) # pole angular velocity

# plt.figure()
# plt.title("Vector Field of Koopman Predicted State Evolution")
# Q_plot = plt.quiver(X_plot, Y_plot, U_plot, V_plot)
# plt.show()

plt.plot(np.mean(norms, axis=0))
plt.title(title)
plt.ylabel('L2 Norm')
plt.xlabel('Timestep')
plt.show()
# %%
