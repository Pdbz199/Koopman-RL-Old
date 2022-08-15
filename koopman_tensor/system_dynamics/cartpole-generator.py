#%% Imports
# import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from scipy import integrate

np.random.seed(123)

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables

#%% True System Dynamics
state_dim = 4
action_dim = 1

state_order = 3
action_order = 3

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

phi_dim = int( sp.special.comb( state_order+state_dim, state_order ) )
psi_dim = int( sp.special.comb( action_order+action_dim, action_order ) )

phi_column_shape = [phi_dim, 1]

action_range = np.array([-25, 25])
step_size = 0.1
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

m =  1  # pendulum mass
M =  5  # cart mass
L =  2  # pendulum length
g = -10 # gravitational acceleration
d =  1  # (delta) cart damping

b = 1   # pendulum up (b = 1)

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
Q = np.eye(4) # state cost matrix, 4x4 identity matrix
R = 0.0001    # control cost matrix

def pendcart(x,t,m,M,L,g,d,uf):
    u = uf(x) # evaluate anonymous control function at x
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx

#%% Policy function(s)
def zero_policy(x):
    return np.array([[0.0]])

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

#%% Rest of dynamics
dt = 0.01
t_span = np.array([0, dt])

def continuous_f(action=None):
    """
        INPUTS:
            action - action column vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
                input - state vector
                t - timestep
        """
        
        uf = lambda x: action
        if uf(0) is None:
            uf = random_policy

        return pendcart(input, t, m, M, L, g, d, uf)

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """

    soln = integrate.solve_ivp(fun=continuous_f(action), t_span=t_span, y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

#%% Generate dataset
num_episodes = 500
num_steps_per_episode = int(20.0 / dt)
N = num_episodes*num_steps_per_episode

X = np.zeros([state_dim,  N])
Y = np.zeros([state_dim,  N])
U = np.zeros([action_dim, N])

for episode in range(num_episodes):
    perturbation = np.array([
        [0],
        [0],
        [np.random.normal(0, 0.05)],
        [0]
    ])
    x = x0 + perturbation

    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        # u = zero_policy(x)
        u = random_policy(x)
        U[:,(episode*num_steps_per_episode)+step] = u
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

#%% Koopman Tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols',
    is_generator=True,
    dt=dt
)

#%% Koopman dynamics
def continuous_tensor_f(action=None):
    """
        INPUTS:
            action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
                t - timestep
                input - state vector
        """

        u = action
        if u is None:
            u = random_policy(input)

        phi_x_column_vector = np.zeros(phi_column_shape)
        for i in range(state_dim):
            phi_x_column_vector[i,0] = input[i]

        phi_x_dot_column_vector = tensor.K_(u) @ phi_x_column_vector

        phi_x_dot = np.zeros(phi_dim)
        for i in range(phi_dim):
            phi_x_dot[i] = phi_x_dot_column_vector[i,0]

        return phi_x_dot

    return f_u

def tensor_f(state, action):
    """
        INPUTS:
            state - state column vector
            action - action column vector

        OUTPUTS:
            state column vector pushed forward in time
    """

    soln = integrate.solve_ivp(fun=continuous_tensor_f(action), t_span=t_span, y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

#%% Plot dynamics
#%% Discretized
steps = int(5.0 / dt)

true_xs = np.zeros([steps,state_dim])
learned_xs = np.zeros([steps,state_dim])

perturbation = np.array([
    [0],
    [0],
    [np.random.normal(0, 0.05)],
    [0]
])
true_x = x0 + perturbation
learned_x = x0 + perturbation
for i in range(steps):
    true_u = zero_policy(true_x)
    # true_u = random_policy(true_x)
    learned_u = true_u

    true_x = f(true_x, true_u)
    true_xs[i] = true_x[:,0]

    learned_x = tensor.B.T @ tensor_f(tensor.phi(learned_x), learned_u)
    learned_xs[i] = learned_x[:,0]

#%% Plot true vs. learned dynamics over time
fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')

axs[0].set_title('True dynamics')
axs[0].set(xlabel='Step', ylabel='State value')

axs[1].set_title('Learned dynamics')
axs[1].set(xlabel='Step', ylabel='State value')

labels = np.array(['cart position', 'cart velocity', 'pole angle', 'pole angular velocity'])
for i in range(state_dim):
    axs[0].plot(true_xs[:,i], label=labels[i])
    axs[1].plot(learned_xs[:,i], label=labels[i])
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()

#%% Define cost
w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    return mat