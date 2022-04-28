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
all_u = np.arange(u_range[0], u_range[1], 0.001)
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
u_dim = 1
order = 2
M_plus_N_minus_ones = np.arange( (x_dim-1), order + (x_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * (x_dim-1) ) ) )
N = 20000 # Number of datapoints
X = np.zeros([x_dim,N])
Y = np.zeros([x_dim,N])
U = np.zeros([u_dim,N])

num_episodes = 200
num_steps_per_episode = 100
for episode in range(num_episodes):
    x = np.zeros([x_dim,1])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.array([[np.random.choice(all_u)]])
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

#%%
model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, all_u.shape[0]),
    torch.nn.Softmax(dim=0)
)

#%%