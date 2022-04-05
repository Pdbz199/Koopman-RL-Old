#%% Imports
# import gym
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

from control.matlab import lqr
from scipy import integrate

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import algorithmsv2_parallel as algorithmsv2
import observables
# import utilities

#%% True System Dynamics
m = 1 # pendulum mass
M = 5 # cart mass
L = 2 # pendulum length
g = -10 # gravitational acceleration
d = 1 # (delta) cart damping

b = 1 # pendulum up (b = 1)

A = np.array([
    [0,1,0,0],
    [0,-d/M,b*m*g/M,0],
    [0,0,0,1],
    [0,- b*d/(M*L),-b*(m+M)*g/(M*L),0]
])
B = np.array([
    [0],
    [1/M],
    [0],
    [b/(M*L)]
])
Q = np.eye(4) # state cost, 4x4 identity matrix
R = 0.0001 # control cost

# This is continuous LQR so Ax + Bu won't work
# def f(x, u):
#     return A @ x + B @ u

def pendcart(x,t,m,M,L,g,d,uf):
    u = uf(x) # evaluate anonymous function at x
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx

#%% Traditional LQR
K = lqr(A, B, Q, R)[0]

#%%
x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

perturbation = np.array([
        [0],
        [0],
        [np.random.normal(0, 0.1)], # np.random.normal(0, 0.05)
        [0]
])
x = x0 + perturbation

w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])

action_range = np.arange(-500, 500)
def random_policy(x):
    return np.array([[np.random.choice(action_range)]])

def optimal_policy(x):
    return -K @ (x - w_r[:, 0])

#%%
# tspan = np.arange(0, 10, 0.001)
# _x = integrate.odeint(pendcart, x[:, 0], tspan, args=(m, M, L, g, d, optimal_policy))

# for i in range(4):
#     plt.plot(_x[:, i])
# plt.show()

#%% Define Simple Dynamics Function
seconds_per_step = 0.002
timespan = np.arange(0, seconds_per_step, 0.001)
def f(x, u):
    policy = lambda state: u
    _x = integrate.odeint(pendcart, x[:, 0], timespan, args=(m, M, L, g, d, policy))
    return np.vstack(_x[-1])

#%% Construct Datasets
num_episodes = 1000
num_steps_per_episode = 300

X = np.zeros([4, num_episodes*num_steps_per_episode])
Y = np.zeros([4, num_episodes*num_steps_per_episode])
U = np.zeros([1, num_episodes*num_steps_per_episode])

for episode in range(num_episodes):
    perturbation = np.array([
            [0],
            [0],
            [np.random.normal(0, 0.1)], # np.random.normal(0, 0.05)
            [0]
    ])
    x = x0 + perturbation

    for step in range(num_steps_per_episode):
        X[:, (episode*num_steps_per_episode)+step] = x[:, 0]
        u = random_policy(x)
        U[:, (episode*num_steps_per_episode)+step] = u
        y = f(x, u)
        Y[:, (episode*num_steps_per_episode)+step] = y[:, 0]
        x = y

#%% Koopman Tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(2),
    regressor='ols'
)

#%% Define cost
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    mat = np.vstack(np.diag((x.T-w_r.T) @ Q @ (x-w_r))) + np.power(u, 2)*R
    return mat

#%% Learn control
u_bounds = np.array([[-500, 500]])
All_U = np.array([np.arange(u_bounds[0,0], u_bounds[0,1], 0.1)])
gamma = 0.99
lamb = 0.1
lr = 1e-1
epsilon = 1e-3

algos = algorithmsv2.algos(
    X,
    All_U,
    u_bounds[0],
    tensor,
    cost,
    gamma=gamma,
    epsilon=epsilon,
    bellman_error_type=0,
    learning_rate=lr,
    weight_regularization_bool=True,
    weight_regularization_lambda=lamb,
    optimizer='adam'
)

algos.w = np.load('bellman-weights.npy')
print("Weights before updating:", algos.w)
# bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
# print("Weights after updating:", algos.w)

#%% Extract policy
All_U_range = np.arange(All_U.shape[1])
def learned_policy(x):
    pis = algos.pis(x)[:,0] # TODO: Check if x needs to be column or row vector
    # Select action column at index sampled from policy distribution
    u_ind = np.random.choice(All_U_range, p=pis)
    u = np.vstack(All_U[:,u_ind])[0,0]
    return u

#%% Test policy by simulating system
num_episodes = 100
num_steps_per_episode = 300

episode_rewards = np.zeros([num_episodes])

for episode in range(num_episodes):
    perturbation = np.array([
            [0],
            [0],
            [np.random.normal(0, 0.1)], # np.random.normal(0, 0.05)
            [0]
    ])
    x = x0 + perturbation

    done = False
    while not done:
        u = learned_policy(x)
        y = f(x, u)
        episode_rewards[episode] += 1.0

        done = y[1,0] > np.pi/2 and y[1,0] < 3*np.pi/2

        x = y

print("Mean reward per episode:", np.mean(episode_rewards))

#%%