#%% Imports
import gym
env = gym.make('env:CartPoleControlEnv-v0')
import matplotlib.pyplot as plt
import numpy as np

from control import dlqr

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% System dynamics
# Cartpole A, B, Q, and R matrices from Wen's homework
A = np.array([
    [1.0, 0.02,  0.0,        0.0 ],
    [0.0, 1.0,  -0.01434146, 0.0 ],
    [0.0, 0.0,   1.0,        0.02],
    [0.0, 0.0,   0.3155122,  1.0 ]
])
B = np.array([
    [0],
    [0.0195122],
    [0],
    [-0.02926829]
])

def f(x, u):
    return A @ x + B @ u

#%% Cost
Q = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1

#%% Traditional LQR solution
C = dlqr(A, B, Q, R)[0]

#%% Construct snapshots of data
action_range = 25
state_range = 25

# step_size = 0.01
# all_us = np.arange(-state_range, state_range+step_size, step_size)

num_episodes = 300
num_steps_per_episode = 500
N = num_episodes * num_steps_per_episode

# Shotgun approach
# X = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
# # U = np.random.rand(B.shape[1],N)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],N))
# # U = np.random.choice(all_us, size=[B.shape[1],N])
# U = -(C @ X)
# Y = f(X, U)

# Path-based approach
X = np.zeros([A.shape[1],num_episodes*num_steps_per_episode])
U = np.zeros([B.shape[1],num_episodes*num_steps_per_episode])
Y = np.zeros([A.shape[1],num_episodes*num_steps_per_episode])
for episode in range(num_episodes):
    state = np.vstack(env.reset())
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = state[:,0]
        action = np.random.rand(B.shape[1],1)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],1))
        # action = np.random.choice(all_us, size=[B.shape[1],1])
        # action = -(C @ state)
        U[:,(episode*num_steps_per_episode)+step] = action[:,0]
        y = f(state, action)
        Y[:,(episode*num_steps_per_episode)+step] = y[:,0]
        state = y

#%% Estimate Koopman tensor
state_order = 2
action_order = 2
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Shotgun-based training error
# training_norms = np.zeros([X.shape[1]])
# state_norms = np.zeros([X.shape[1]])
# for i in range(X.shape[1]):
#     state = np.vstack(X[:,i])
#     state_norms[(episode*num_steps_per_episode)+step] = utilities.l2_norm(state, np.zeros_like(state))
#     action = np.vstack(U[:,i])
#     true_x_prime = np.vstack(Y[:,i])
#     predicted_x_prime = tensor.f(state, action)
#     training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
# average_training_norm = np.mean(training_norms)
# average_state_norm = np.mean(state_norms)
# print(f"Average training norm: {average_training_norm}")
# print(f"Average training norm normalized by average state norm: {average_training_norm / average_state_norm}")

#%% Path-based training error
training_norms = np.zeros([num_episodes,num_steps_per_episode])
state_norms = np.zeros([X.shape[1]])
for episode in range(num_episodes):
    for step in range(num_steps_per_episode):
        state = np.vstack(X[:,(episode*num_steps_per_episode)+step])
        state_norms[(episode*num_steps_per_episode)+step] = utilities.l2_norm(state, np.zeros_like(state))
        action = np.vstack(U[:,(episode*num_steps_per_episode)+step])
        true_x_prime = np.vstack(Y[:,(episode*num_steps_per_episode)+step])
        predicted_x_prime = tensor.f(state, action)
        training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
        state = true_x_prime
average_training_norm_per_episode = np.mean(np.sum(training_norms, axis=1))
average_state_norm = np.mean(state_norms)
print(f"Average training norm per episode over {num_episodes} episodes: {average_training_norm_per_episode}")
print(f"Average training norm per episode over {num_episodes} episodes normalized by average state norm: {average_training_norm_per_episode / average_state_norm}")

#%% Plot state path
num_steps = 500
true_states = np.zeros([num_steps, A.shape[1]])
true_actions = np.zeros([num_steps, B.shape[1]])
koopman_states = np.zeros([num_steps, A.shape[1]])
koopman_actions = np.zeros([num_steps, B.shape[1]])
state = np.vstack(env.reset())
true_state = state
koopman_state = state
for i in range(num_steps):
    true_states[i] = true_state[:,0]
    koopman_states[i] = koopman_state[:,0]
    # true_action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1]), size=(1,1))
    # true_action = np.random.choice(all_us, size=[B.shape[1],1])
    true_action = -(C @ true_state)
    # koopman_action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1]), size=(1,1))
    # koopman_action = np.random.choice(all_us, size=[B.shape[1],1])
    koopman_action = -(C @ koopman_state)
    true_state = f(true_state, true_action)
    koopman_state = tensor.f(koopman_state, koopman_action)
print("Norm between entire paths:", utilities.l2_norm(true_states, koopman_states))

fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')

axs[0].set_title('True dynamics')
axs[0].set(xlabel='Timestep', ylabel='State value')

axs[1].set_title('Learned dynamics')
axs[1].set(xlabel='Timestep', ylabel='State value')

labels = np.array(['cart position', 'cart velocity', 'pole angle', 'pole angular velocity'])
for i in range(A.shape[1]):
    axs[0].plot(true_states[:,i], label=labels[i])
    axs[1].plot(koopman_states[:,i], label=labels[i])
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()

#%%
