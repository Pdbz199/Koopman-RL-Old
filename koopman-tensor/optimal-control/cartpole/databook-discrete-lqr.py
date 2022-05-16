#%% Imports
import gym
env = gym.make('env:CartPoleControlEnv-v0')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from control import dlqr
from scipy.integrate import quad_vec

import sys
sys.path.append('../../')
from tensor import KoopmanTensor
sys.path.append('../../../')
import observables
import utilities

#%% True System Dynamics From Databook
m = 1 # pendulum mass
M = 5 # cart mass
L = 2 # pendulum length
g = -10 # gravitational acceleration
d = 1 # (delta) cart damping

b = 1 # pendulum up (b = 1)

continuous_A = np.array([
    [0,1,0,0],
    [0,-d/M,b*m*g/M,0],
    [0,0,0,1],
    [0,- b*d/(M*L),-b*(m+M)*g/(M*L),0]
])
delta_t = 0.002
A = sp.linalg.expm(continuous_A * delta_t)
continuous_B = np.array([
    [0],
    [1/M],
    [0],
    [b/(M*L)]
])
B = A @ continuous_B

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

# print(A)
# print(np.linalg.eig(A))
# print(B)

def f(x, u):
    return (A @ x + B @ u)

#%% Cost
# Q and R from Databook
Q = np.eye(4) # state cost, 4x4 identity matrix
R = 0.0001 # control cost

# Q and R from Wen's homework
Q = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1

# Reference point from Databook
# w_r = np.array([
#     [1],
#     [0],
#     [np.pi],
#     [0]
# ])

# Reference point from Wen's homework
w_r = np.zeros([A.shape[1],1])

def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    x_r = x - w_r
    return np.vstack(np.diag(x_r.T @ Q @ x_r)) + np.power(u, 2)*R

# print(Q)
# print(R)

#%% Traditional LQR solution
C = dlqr(A, B, Q, R)[0]

#%% Initial x
x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

#%% Construct snapshots of data
action_range = 25
state_range = 25

# step_size = 0.01
# all_us = np.arange(-state_range, state_range+step_size, step_size)

# Shotgun approach
# N = 20000
# U = np.random.rand(B.shape[1],N)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],N))
# U = np.random.choice(all_us, size=[B.shape[1],N])
# X = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
# Y = f(X, U)

# Path-based approach
num_episodes = 20000
num_steps_per_episode = 100
X = np.zeros([A.shape[1],num_episodes*num_steps_per_episode])
U = np.zeros([B.shape[1],num_episodes*num_steps_per_episode])
Y = np.zeros([A.shape[1],num_episodes*num_steps_per_episode])
for episode in range(num_episodes):
    state = np.vstack(env.reset())
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = state[:,0]
        action = np.random.rand(B.shape[1],1)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],1))
        # action = np.random.choice(all_us, size=[B.shape[1],1])
        # action = -(C @ (state - w_r))
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

#%% Training error
training_norms = np.zeros([num_episodes,num_steps_per_episode])
for episode in range(num_episodes):
    state = np.vstack(X[:,episode*num_steps_per_episode])
    for step in range(num_steps_per_episode):
        action = np.vstack(U[:,(episode*num_steps_per_episode)+step])
        true_x_prime = np.vstack(Y[:,(episode*num_steps_per_episode)+step])
        predicted_x_prime = tensor.f(state, action)
        training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
        state = np.vstack(X[:,(episode*num_steps_per_episode)+step])
    
print(f"Average training norm per step over {num_episodes} episodes:", np.mean(np.sum(training_norms, axis=1)))

#%% Plot state path
num_steps = 1000
true_states = np.zeros([num_steps, A.shape[1]])
true_actions = np.zeros([num_steps, B.shape[1]])
koopman_states = np.zeros([num_steps, A.shape[1]])
koopman_actions = np.zeros([num_steps, B.shape[1]])
# perturbation = np.array([
#     [0],
#     [0],
#     [np.random.normal(0, 0.05)],
#     [0]
# ])
# state = x0 + perturbation
state = np.vstack(env.reset())
true_state = state
koopman_state = state
for i in range(num_steps):
    true_states[i] = true_state[:,0]
    koopman_states[i] = koopman_state[:,0]
    # true_action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1]), size=(1,1))
    # true_action = np.random.choice(all_us, size=[B.shape[1],1])
    true_action = -(C @ (true_state - w_r))
    # koopman_action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1]), size=(1,1))
    # koopman_action = -(C @ (koopman_state - w_r))
    true_state = f(true_state, true_action)
    koopman_state = tensor.f(koopman_state, true_action)

fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')
axs[0].set_title('True dynamics')
axs[1].set_title('Learned dynamics')
print("Norm between entire paths:", utilities.l2_norm(true_states, koopman_states))
for i in range(A.shape[1]):
    axs[0].plot(true_states[:,i])
    axs[1].plot(koopman_states[:,i])
plt.show()

#%%
