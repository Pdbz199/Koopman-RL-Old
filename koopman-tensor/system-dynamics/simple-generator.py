# import gym
import numpy as np
import scipy as sp

seed = 123
np.random.seed(seed)

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables

#%% System dynamics
state_dim = 2
action_dim = 1

state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

action_range = np.array([-50, 50])
# action_range = np.array([-75, 75])
step_size = 0.1
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

gamma = -0.8
delta = -0.7
t_span = np.arange(0, 0.01, 0.001)

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
        x_1, x_2 = input

        x_1_dot = gamma * x_1
        x_2_dot = delta * (x_2 - x_1**2)

        u = action
        if u is None:
            u = np.random.choice(all_actions, size=action_column_shape)

        return [ x_1_dot, x_2_dot ]

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

#%% Generate data
num_episodes = 500
num_steps_per_episode = 1000
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

# initial_xs = np.zeros([num_episodes, state_dim])
# for episode in range(num_episodes):
#     x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
#     u = np.array([[0]])
#     soln = solve_ivp(fun=continuous_f(u), t_span=[0.0, 10.0], y0=x[:,0], method='RK45')
#     initial_xs[episode] = soln.y[:,-1]

initial_xs = np.random.rand(num_episodes,state_dim) * 2 * np.random.choice([-1,1], size=[num_episodes,state_dim])

for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        # u = np.random.choice(all_actions, size=action_column_shape)
        u = np.array([[0]])
        U[:,(episode*num_steps_per_episode)+step] = u[:,0]
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    is_generator=True,
    regressor='ols'
)

#%% Policy function(s)
def zero_policy(x):
    return np.array([[0]])

def random_policy(x):
    return np.random.choice(action_range, size=action_column_shape)

#%% Compute true and learned dynamics over time
initial_state = np.random.rand(*state_column_shape) * 2 * np.random.choice([-1,1], size=state_column_shape)
tspan = np.arange(0, 25, 0.001)
u = zero_policy(initial_state)

true_xs = odeint(continuous_f(u), initial_state[:,0], tspan, tfirst=True)
# learned_xs = odeint(continuous_tensor_f(u), tensor.phi(initial_state)[:,0], tspan, tfirst=True)
learned_xs = np.zeros([25000,phi_dim])
phi_state = tensor.phi(initial_state)
for i in range(learned_xs.shape[0]):
    # learned_xs[i] = (sp.linalg.expm((0.1*(i+1)) * tensor.K_(u)) @ tensor.phi(initial_state))[:,0]
    learned_xs[i] = (sp.linalg.expm(0.1 * tensor.K_(u)) @ phi_state)[:,0]
    phi_state = np.vstack(learned_xs[i])
learned_xs = (tensor.B.T @ learned_xs.T).T

#%% Plot true vs. learned dynamics over time
fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')

axs[0].set_title('True dynamics')
axs[0].set(xlabel='Step', ylabel='State value')

axs[1].set_title('Learned dynamics')
axs[1].set(xlabel='Step', ylabel='State value')

labels = np.array(['x_1', 'x_2'])
for i in range(state_dim):
    axs[0].plot(true_xs[:,i], label=labels[i])
    axs[1].plot(learned_xs[:,i], label=labels[i])
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()