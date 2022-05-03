#%% Imports
import numpy as np
import scipy as sp
import torch

seed = 123
np.random.seed(seed)

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, odeint

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% System dynamics
x_dim = 3
x_column_shape = [x_dim, 1]
u_dim = 1
u_column_shape = [u_dim, 1]
order = 2
M_plus_N_minus_ones = np.arange( (x_dim-1), order + (x_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * ( x_dim-1 ) ) ) )

u_range = np.array([-1, 1])
all_us = np.arange(u_range[0], u_range[1], 0.01) #* if too compute heavy, 0.05
omega = 1.0
mu = 0.1
A = -0.1
lamb = 1
# t_span = np.arange(0, 0.02, 0.001)
t_span = np.arange(0, 0.001, 0.0001)

# def continuous_f(input, t, u):
#     """
#         INPUTS:
#         input - state vector
#         t - timestep
#         u - action vector
#     """
#     x, y, z = input

#     x_dot = mu*x - omega*y + A*x*z
#     y_dot = omega*x + mu*y + A*y*z
#     z_dot = -lamb * (z - np.power(x, 2) - np.power(y, 2))

#     return [x_dot, y_dot + u, z_dot]

def continuous_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - state vector
            t - timestep
        """
        x, y, z = input

        x_dot = mu*x - omega*y + A*x*z
        y_dot = omega*x + mu*y + A*y*z
        z_dot = -lamb * ( z - np.power(x, 2) - np.power(y, 2) )

        u = action
        if u is None:
            u = np.random.choice(all_us, size=u_column_shape)

        return [ x_dot, y_dot + u, z_dot ]

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

#%% Graph sample state over time
# state = np.array([
#     [0.0], # x
#     [0.1], # y
#     [1.0]  # z
# ])
# integration = odeint(func=continuous_f([0]), y0=state[:,0], t=np.arange(0, 100, 0.001), tfirst=True)
# ax = plt.axes(projection='3d')
# ax.set_xlim(-1.0, 1.0)
# ax.set_ylim(-1.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.plot3D(integration[:,0], integration[:,1], integration[:,2], 'gray')
# plt.show()

#%%
num_episodes = 500
num_steps_per_episode = 1000
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([x_dim,N])
Y = np.zeros([x_dim,N])
U = np.zeros([u_dim,N])

# state = np.array([
#     [0.0], # x
#     [0.1], # y
#     [1.0]  # z
# ])
# for step in range(num_steps_per_episode):
#     X[:,step] = state[:,0]
#     action = np.random.choice(all_us, size=u_column_shape)
#     U[:,step] = action[:,0]
#     state = f(state, action)
# ax = plt.axes(projection='3d')
# ax.set_xlim(-1.0, 1.0)
# ax.set_ylim(-1.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.plot3D(
#     X[0,:num_steps_per_episode],
#     X[1,:num_steps_per_episode],
#     X[2,:num_steps_per_episode],
#     'gray'
# )
# plt.show()

initial_xs = np.zeros([num_episodes, x_dim])
for episode in range(num_episodes):
    x = np.random.random(x_column_shape) * 0.5 * np.random.choice([-1,1], size=x_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 10.0], y0=x[:,0], method='RK45')
    initial_xs[episode] = soln.y[:,-1]

for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    # x = np.random.rand(x_dim,1)
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.random.choice(all_us, size=u_column_shape)
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

# state = np.array([
#     [0.0], # x
#     [0.1], # y
#     [1.0]  # z
# ])
# num_steps_per_episode = 100000
# for step in range(num_steps_per_episode):
#     X[:,step] = state[:,0]
#     # action = np.random.choice(all_us, size=u_column_shape)
#     action = np.array([[0.5]])
#     U[:,step] = action[:,0]
#     state = tensor.f(state, action)
# ax = plt.axes(projection='3d')
# ax.set_xlim(-1.0, 1.0)
# ax.set_ylim(-1.0, 1.0)
# ax.set_zlim(0.0, 1.0)
# ax.plot3D(
#     X[0,:num_steps_per_episode],
#     X[1,:num_steps_per_episode],
#     X[2,:num_steps_per_episode],
#     'gray'
# )

#%% Training error
training_norms = np.zeros([num_episodes, num_steps_per_episode])
for episode in range(num_episodes):
    for step in range(num_steps_per_episode):
        x = np.vstack(X[:,(episode*num_steps_per_episode)+step])
        u = np.vstack(U[:,(episode*num_steps_per_episode)+step])

        true_x_prime = np.vstack(
            Y[:,(episode*num_steps_per_episode)+step]
        )
        predicted_x_prime = tensor.f(x, u)

        training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
print(f"Mean training norm per episode over {num_episodes} episodes:", np.mean( np.sum(training_norms, axis=1) ))

#%% Testing error
testing_norms = np.zeros([num_episodes, num_steps_per_episode])
for episode in range(num_episodes):
    # x = np.vstack(initial_xs[episode])
    # x = np.random.random(x_column_shape) * np.random.choice([-1,1], size=x_column_shape)
    init_x = np.random.random(x_column_shape) * 0.5 * np.random.choice([-1,1], size=x_column_shape)
    soln = solve_ivp(fun=continuous_f(np.array([[0]])), t_span=[0, t_span[-1]*num_steps_per_episode], y0=init_x[:,0], method='RK45')
    x = np.vstack(soln.y[:,-1])

    for step in range(num_steps_per_episode):
        u = np.random.choice(all_us, size=u_column_shape)

        true_x_prime = f(x, u)
        predicted_x_prime = tensor.f(x, u)

        testing_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
print(f"Mean testing norm per episode over {num_episodes} episodes:", np.mean( np.sum(testing_norms, axis=1) ))

#%%
model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, all_us.shape[0])#,
    # torch.nn.Softmax(dim=0)
)

#%%