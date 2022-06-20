# import gym
import numpy as np

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
state_dim = 3
action_dim = 1

state_order = 4
action_order = 4

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

action_range = np.array([-50, 50])
step_size = 0.1
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

sigma = 10
rho = 28
beta = 8/3
t_span = np.arange(0, 0.01, 0.001)

x_e = np.sqrt( beta * ( rho - 1 ) )
y_e = np.sqrt( beta * ( rho - 1 ) )
z_e = rho - 1

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

        x_dot = sigma * ( y - x )
        y_dot = ( rho - z ) * x - y
        z_dot = x * y - beta * z

        u = action
        if u is None:
            u = np.random.choice(all_actions, size=action_column_shape)

        return [ x_dot + u, y_dot, z_dot ]

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

for episode in range(num_episodes):
    # x = np.vstack(initial_xs[episode])
    x = np.array([[0], [1], [1.05]]) + np.random.rand(*state_column_shape) #* np.random.choice([-1,1], size=state_column_shape)
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

#%% Construct continuous dynamics from tensor
def continuous_tensor_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - phi state vector
            t - timestep
        """

        u = action
        if u is None:
            u = np.random.choice(all_actions, size=action_column_shape)

        phi_state_column_vector = np.zeros([phi_dim,1])
        for i in range(phi_dim):
            phi_state_column_vector[i,0] = input[i]

        phi_x_prime_computation = tensor.K_(u) @ phi_state_column_vector

        phi_x_prime = np.zeros(phi_dim) # is this phi_x_prime or is it phi_x_dot_prime?
        for i in range(phi_dim):
            phi_x_prime[i] = phi_x_prime_computation[i,0]

        return phi_x_prime

    return f_u

#%% Policy function(s)
def zero_policy(x):
    return np.array([[0]])

def random_policy(x):
    return np.random.choice(action_range, size=action_column_shape)

#%% Compute true and learned dynamics over time
initial_state = np.array([[0], [1], [1.05]])
tspan = np.arange(0, 25, 0.001)
u = np.array([0.0])

true_xs = odeint(continuous_f(u), initial_state[:,0], tspan, tfirst=True)
learned_xs = odeint(continuous_tensor_f(u), tensor.phi(initial_state)[:,0], tspan, tfirst=True)
learned_xs = (tensor.B.T @ learned_xs.T).T

#%% Plot true vs. learned dynamics over time
fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')

axs[0].set_title('True dynamics')
axs[0].set(xlabel='Step', ylabel='State value')

axs[1].set_title('Learned dynamics')
axs[1].set(xlabel='Step', ylabel='State value')

labels = np.array(['x_0', 'x_1', 'x_2'])
for i in range(state_dim):
    axs[0].plot(true_xs[:,i], label=labels[i])
    axs[1].plot(learned_xs[:,i], label=labels[i])
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()