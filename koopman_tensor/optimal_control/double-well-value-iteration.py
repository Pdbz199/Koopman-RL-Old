#%% Imports
from tkinter import W
import numpy as np
import torch
import torch.nn as nn

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import care
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

PATH = './double-well-value-model.pt'

#%% System details
state_dim = 2
action_dim = 1

state_order = 2 # 3?
action_order = 2 # 3?

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

phi_column_shape = [phi_dim, 1]

action_range = np.array([-75, 75])
step_size = 1.0
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

#%% System dynamics
gamma = 0.99
lamb = 1

dt = 0.01
t_span = np.arange(0, dt, dt/10)

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
        x, y = input
        
        u = action
        if u is None:
            u = random_policy(x_dot)

        b_x = np.array([
            [4*x - 4*(x**3)],
            [-2*y]
        ]) + u
        sigma_x = np.array([
            [0.7, x],
            [0, 0.5]
        ])

        column_output = b_x + sigma_x * np.random.randn(2,1)
        x_dot = column_output[0,0]
        y_dot = column_output[1,0]

        return [ x_dot, y_dot ]

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

#%% Reward function

# LQR
w_r = np.array([
    [0.0],
    [0.0],
])
Q_ = np.eye(state_dim)
R = 0.001
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    # x.T @ Q @ x + u.T @ R @ u
    return mat.T
def reward(x, u):
    return -cost(x, u)

#%% Default policy functions
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

#%% Generate data
num_episodes = 500
num_timesteps = 10.0
num_steps_per_episode = int(num_timesteps / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_x = np.array([[-0.5], [0.7]]) # Picked out of a hat

for episode in range(num_episodes):
    x = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        # u = zero_policy(x)
        u = random_policy(x)
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
    regressor='ols'
)


#%% Policy function as PyTorch model
def inner_pi_us(us, xs):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]

    V_x_primes_arr = torch.zeros([all_actions.shape[0], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

    inner_pi_us_values = -(torch.from_numpy(cost(xs, us)).float() + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = np.finfo(np.float32).eps # 1e-25

    inner_pi_us_response = torch.real(inner_pi_us(np.array([all_actions]), xs)) # all_actions.shape[0] x xs.shape[1]

    # Max trick
    max_inner_pi_u = torch.amax(inner_pi_us_response, axis=0) # xs.shape[1]
    diff = inner_pi_us_response - max_inner_pi_u

    pi_us = torch.exp(diff) + delta # all_actions.shape[0] x xs.shape[1]
    Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]
    
    return pi_us / Z_x # all_actions.shape[0] x xs.shape[1]

def discrete_bellman_error(batch_size):
    """ Equation 12 in writeup """
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # X.shape[0] x batch_size
    phi_xs = tensor.phi(x_batch) # dim_phi x batch_size
    phi_x_primes = tensor.K_(np.array([all_actions])) @ phi_xs # all_actions.shape[0] x dim_phi x batch_size

    pis_response = pis(x_batch) # all_actions.shape[0] x x_batch_size
    log_pis = torch.log(pis_response) # all_actions.shape[0] x batch_size

    # Compute V(x)'s
    V_x_primes_arr = torch.zeros([all_actions.shape[0], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T
    
    # Get costs
    costs = torch.from_numpy(cost(x_batch, np.array([all_actions]))).float() # all_actions.shape[0] x batch_size

    # Compute expectations
    expectation_us = (costs + lamb*log_pis + gamma*V_x_primes_arr) * pis_response # all_actions.shape[0] x batch_size
    expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

    # Use model to get V(x) for all phi(x)s
    V_xs = model(torch.from_numpy(phi_xs.T).float()) # (batch_size, 1)

    # Compute squared differences
    squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
    total = torch.sum(squared_differences) / batch_size # scalar

    return total

#%%
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

# model = torch.nn.Sequential(torch.nn.Linear(phi_dim, 1))
# model.apply(init_weights)

# learning_rate = 0.003
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model = torch.load(PATH)

#%% Run Algorithm
# epochs = 500
epochs = 0
epsilon = 1e-2
batch_size = 2**9
batch_scale = 3
bellman_errors = [discrete_bellman_error(batch_size*batch_scale).data.numpy()]
BE = bellman_errors[-1]
print("Initial Bellman error:", BE)

while gamma <= 0.99:
    for epoch in range(epochs):
        # Get random batch of X and Phi_X
        x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
        x_batch = X[:,x_batch_indices] # X.shape[0] x batch_size
        phi_x_batch = tensor.phi(x_batch) # dim_phi x batch_size

        # Compute estimate of V(x) given the current model
        V_x = model(torch.from_numpy(phi_x_batch.T).float()).T # (1, batch_size)

        # Get current distribution of actions for each state
        pis_response = pis(x_batch) # (all_actions.shape[0], batch_size)
        log_pis = torch.log(pis_response) # (all_actions.shape[0], batch_size)

        # Compute V(x)'
        phi_x_primes = tensor.K_(np.array([all_actions])) @ phi_x_batch # all_actions.shape[0] x dim_phi x batch_size
        V_x_primes_arr = torch.zeros([all_actions.shape[0], batch_size])
        for u in range(phi_x_primes.shape[0]):
            V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T

        # Get costs
        costs = torch.from_numpy(cost(x_batch, np.array([all_actions]))).float() # (all_actions.shape[0], batch_size)

        # Compute expectations
        expectation_term_1 = torch.sum(
            torch.mul(
                (costs + lamb*log_pis + gamma*V_x_primes_arr),
                pis_response
            ),
            dim=0
        ).reshape(1,-1) # (1, batch_size)

        # Equation 2.21 in Overleaf
        loss = torch.sum( torch.pow( V_x - expectation_term_1, 2 ) ) # ()
        
        # Back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Recompute Bellman error
        BE = discrete_bellman_error(batch_size*batch_scale).data.numpy()
        bellman_errors = np.append(bellman_errors, BE)

        # Every so often, print out and save the bellman error(s)
        if (epoch+1) % 250 == 0:
            # np.save('double_well_bellman_errors.npy', bellman_errors)
            torch.save(model, PATH)
            print(f"Bellman error at epoch {epoch+1}: {BE}")

        if BE <= epsilon:
            torch.save(model, PATH)
            break

    gamma += 0.02

#%% Extract
def learned_policy(x):
    with torch.no_grad():
        pis_response = pis(x)[:,0]
    return np.random.choice(all_actions, size=action_column_shape, p=pis_response.data.numpy())

#%% Test policy in environment
num_episodes = 100 # for eval avg cost diagnostic

def watch_agent(num_episodes, num_steps_per_episode):
    states = np.zeros([num_episodes,state_dim,num_steps_per_episode])
    actions = np.zeros([num_episodes,action_dim,num_steps_per_episode])
    costs = torch.zeros([num_episodes])
    for episode in range(num_episodes):
        state = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
        for step in range(num_steps_per_episode):
            states[episode,:,step] = state[:,0]
            action = learned_policy(state)
            # action = lqr_policy(state)
            # if action[0,0] > action_range[1]:
            #     action = np.array([[action_range[1]]])
            # if action[0,0] < action_range[0]:
            #     action = np.array([[action_range[0]]])
            actions[episode,:,step] = action
            state = tensor.f(state, action)
            costs[episode] += cost(state, action)[0,0]
    print(f"Mean cost per episode over {num_episodes} episode(s): {torch.mean(costs)}")
    print(f"Initial state of final episode: {states[-1,:,0]}")
    print(f"Final state of final episode: {states[-1,:,-1]}")
    print(f"Reference state: {w_r[:,0]}")
    print(f"Difference between final state of final episode and reference state: {np.abs(states[-1,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of final episode and reference state: {utilities.l2_norm(states[-1,:,-1], w_r[:,0])}")

    ax = plt.axes()
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.plot(states[-1,0], states[-1,1], 'gray')
    plt.show()

    plt.scatter(np.arange(actions.shape[2]), actions[-1,0], s=5)
    plt.show()

    plt.hist(actions[-1,0])
    plt.show()
print("Testing learned policy...")

num_timesteps = 15.0
step_limit = int(num_timesteps / dt)
watch_agent(100, step_limit)

#%%

