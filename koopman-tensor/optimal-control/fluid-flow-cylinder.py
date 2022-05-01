#%% Imports
import numpy as np
import scipy as sp
import torch

from scipy.integrate import odeint

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% System dynamics
u_range = np.array([-1, 1])
all_us = np.arange(u_range[0], u_range[1], 0.01) # 0.001
omega = 1.0
mu = 0.1
A = 0.1
lamb = 1
t_span = np.arange(0, 0.02, 0.001)

def continuous_f(input, t, u):
    """
        INPUTS:
        input - state vector
        t - timestep
        u - action vector
    """
    x, y, z = input

    x_dot = mu*x - omega*y + A*x*z
    y_dot = omega*x + mu*y + A*y*z
    z_dot = -lamb * (z - np.power(x, 2) - np.power(y, 2))

    return [x_dot, y_dot + u, z_dot]

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action[:,0]

    soln = odeint(continuous_f, state[:,0], t_span, args=(u,))
    
    return np.vstack(soln[-1])

#%%
x_dim = 3
x_column_shape = [x_dim, 1]
u_dim = 1
u_column_shape = [u_dim, 1]
order = 2
M_plus_N_minus_ones = np.arange( (x_dim-1), order + (x_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * (x_dim-1) ) ) )
N = 20000 # Number of datapoints
X = np.zeros([x_dim,N])
Y = np.zeros([x_dim,N])
U = np.zeros([u_dim,N])

num_episodes = 200
num_steps_per_episode = 100

initial_xs = np.zeros([num_episodes, x_dim])

for episode in range(num_episodes):
    x = np.random.random(x_column_shape) * 2 * np.random.choice([-1,1], size=x_column_shape)
    for step in range(num_steps_per_episode):
        u = np.random.choice(all_us, size=u_column_shape)
        x = f(x, u)
    initial_xs[episode] = x[:,0]

for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.array([[np.random.choice(all_us)]])
        U[:,(episode*num_steps_per_episode)+step] = u[:,0]
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

#%% Estimate Koopman operator
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(order),
    psi=observables.monomials(order),
    regressor='ols'
)

#%% Training error
training_norms = np.zeros([num_episodes, num_steps_per_episode])
for episode in range(num_episodes):
    x = np.vstack(X[:,episode])
    for step in range(num_steps_per_episode):
        u = np.vstack(U[:,(episode*num_steps_per_episode)+step])

        true_x_prime = np.vstack(Y[:,(episode*num_steps_per_episode)+step])
        predicted_x_prime = tensor.f(x, u)

        training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
print(f"Mean training norm per episode over {num_episodes} episodes:", np.mean( np.sum(training_norms, axis=1) ))

#%% Testing error
testing_norms = np.zeros([num_episodes, num_steps_per_episode])
for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    for step in range(num_steps_per_episode):
        u = np.random.choice(all_us, size=u_column_shape)

        true_x_prime = f(x, u)
        predicted_x_prime = tensor.f(x, u)

        testing_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
print(f"Mean testing norm per episode over {num_episodes} episodes:", np.mean( np.sum(testing_norms, axis=1) ))

#%%
model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, all_us.shape[0]),
    torch.nn.Softmax(dim=0)
)

#%%