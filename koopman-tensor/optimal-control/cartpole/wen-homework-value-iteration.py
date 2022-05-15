#%% Imports
import numpy as np
import random
import torch

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

from matplotlib import pyplot as plt
from scipy.special import comb

import sys
sys.path.append('../../')
from tensor import KoopmanTensor
sys.path.append('../../../')
import cartpole_reward
import observables
import utilities

#%% Initialize environment
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

theta_threshold_radians = 12 * 2 * np.pi / 360
x_threshold = 2.4
def is_done(state):
    """
        INPUTS:
        state - state array
    """
    return bool(
        state[0] < -x_threshold
        or state[0] > x_threshold
        or state[2] < -theta_threshold_radians
        or state[2] > theta_threshold_radians
    )

#%% Initialize important vars
state_order = 2
action_order = 2
state_dim = A.shape[1]
action_dim = B.shape[1]
state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
state_Ns = np.arange( state_dim - 1, state_dim - 1 + (state_order+1) )
phi_dim = int( torch.sum( torch.from_numpy( comb( state_Ns, np.ones_like(state_Ns) * (state_order+1) ) ) ) )
action_Ns = np.arange( action_dim - 1, action_dim - 1 + (action_order+1) )
psi_dim = int( torch.sum( torch.from_numpy( comb( action_Ns, np.ones_like(action_Ns) * (action_order+1) ) ) ) )

step_size = 0.1
all_us = np.arange(-5, 5+step_size, step_size)
all_us = np.round(all_us, decimals=2)

#%% Construct snapshots of u from random agent and initial states x0
N = 20000 # Number of datapoints
X = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])
Y = np.zeros([state_dim,N])
i = 0
while i < N:
    state = np.array([
        [0],
        [0],
        [np.random.normal(0, 0.05)],
        [0]
    ])
    done = False
    step = 0
    while i < N and not done and step < 200:
        X[:,i] = state[:,0]
        U[0,i] = np.random.choice(all_us)
        Y[:,i] = f(state, np.array([[ U[0,i] ]]))[:, 0]
        done = is_done(Y[:,i])
        state = np.vstack(Y[:,i])
        i += 1
        step += 1

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Training error
training_norms = np.zeros([N])
for step in range(N):
    x = np.vstack(X[:,step])
    u = np.vstack(U[:,step])

    true_x_prime = np.vstack(Y[:,step])
    predicted_x_prime = tensor.f(x, u)

    training_norms[step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
print(f"Mean training norm per episode over {N} steps:", np.mean( training_norms ))

#%% Testing error
num_episodes = 200
testing_norms = np.zeros([num_episodes])
for episode in range(num_episodes):
    state = np.array([
        [0],
        [0],
        [np.random.normal(0, 0.05)],
        [0]
    ])
    done = False
    step = 0
    while not done and step < 200:
        action = np.random.choice(all_us, size=action_column_shape)
        true_x_prime = f(state, action)
        done = is_done(true_x_prime[:, 0])
        predicted_x_prime = tensor.f(state, action)

        testing_norms[episode] += utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
        step += 1
print(f"Mean testing norm per episode over {num_episodes} episodes:", np.mean( testing_norms ))

#%% Pytorch setup
model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, 1)
)
# Initialize weights with 0s
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)
model.apply(init_weights)
# torch.load('wen-homework-value-iteration.pt')

gamma = 0.99
lamb = 0.0001
learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% Define cost
# def cost(x, u):
#     return -cartpole_reward.defaultCartpoleRewardMatrix(x, u)

Q_ = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    # x.T Q x + u.T R u
    mat = np.vstack(np.diag(x.T @ Q_ @ x)) + np.power(u, 2)*R
    return mat.T

#%% Control
def inner_pi_us(us, xs):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]

    V_x_primes_arr = torch.zeros([all_us.shape[0], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

    inner_pi_us_values = -(torch.from_numpy(cost(xs, us)).float() + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = 1e-25

    inner_pi_us_response = torch.real(inner_pi_us(np.array([all_us]), xs)) # all_us.shape[0] x xs.shape[1]

    # Max trick
    max_inner_pi_u = torch.amax(inner_pi_us_response, axis=0) # xs.shape[1]
    diff = inner_pi_us_response - max_inner_pi_u

    pi_us = torch.exp(diff) + delta # all_us.shape[0] x xs.shape[1]
    Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]
    
    return pi_us / Z_x # all_us.shape[0] x xs.shape[1]

def discrete_bellman_error(batch_size):
    ''' Equation 12 in writeup '''
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # X.shape[0] x batch_size
    phi_xs = tensor.phi(x_batch) # dim_phi x batch_size
    phi_x_primes = tensor.K_(np.array([all_us])) @ phi_xs # all_us.shape[0] x dim_phi x batch_size

    pis_response = pis(x_batch) # all_us.shape[0] x x_batch_size
    log_pis = torch.log(pis_response) # all_us.shape[0] x batch_size

    # Compute V(x)'s
    V_x_primes_arr = torch.zeros([all_us.shape[0], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T
    
    # Get costs
    costs = torch.from_numpy(cost(x_batch, np.array([all_us]))).float() # all_us.shape[0] x batch_size

    # Compute expectations
    expectation_us = (costs + lamb*log_pis + gamma*V_x_primes_arr) * pis_response # all_us.shape[0] x batch_size
    expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

    # Use model to get V(x) for all phi(x)s
    V_xs = model(torch.from_numpy(phi_xs.T).float()) # (batch_size, 1)

    # Compute squared differences
    squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
    total = torch.sum(squared_differences) / batch_size # scalar

    return total

epochs = 10000
epsilon = 0.00126 # 1e-5
batch_size = 2**13 # 2**9 # 512
bellman_errors = [discrete_bellman_error(batch_size).data.numpy()]
BE = bellman_errors[-1]
print("Initial Bellman error:", BE)

count = 0
# while BE > epsilon:
for _ in range(epochs):
    # Get random batch of X and Phi_X
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:,x_batch_indices] # X.shape[0] x batch_size
    phi_x_batch = tensor.phi(x_batch) # dim_phi x batch_size

    # Compute estimate of V(x) given the current model
    V_x = model(torch.from_numpy(phi_x_batch.T).float()).T # (1, batch_size)

    # Get current distribution of actions for each state
    pis_response = pis(x_batch) # (all_us.shape[0], batch_size)
    log_pis = torch.log(pis_response) # (all_us.shape[0], batch_size)

    # Compute V(x)'
    phi_x_primes = tensor.K_(np.array([all_us])) @ phi_x_batch # all_us.shape[0] x dim_phi x batch_size
    V_x_primes_arr = torch.zeros([all_us.shape[0], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T

    # Get costs
    costs = torch.from_numpy(cost(x_batch, np.array([all_us]))).float() # (all_us.shape[0], batch_size)

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
    BE = discrete_bellman_error(batch_size).data.numpy()
    bellman_errors = np.append(bellman_errors, BE)

    # Every so often, print out and save the bellman error(s)
    if count % 500 == 0:
        np.save('bellman_errors.npy', bellman_errors)
        torch.save(model, 'wen-homework-value-iteration.pt')
        print("Current Bellman error:", BE)

    count += 1

    if BE <= epsilon:
        break

#%% Extract latest policy
def policy(x):
    pis_response = pis(x)[:,0]
    return np.random.choice(all_us, p=pis_response.data.numpy())

#%% Test learned policy
num_episodes = 1000
rewards = np.zeros([num_episodes])
for episode in range(num_episodes):
    state = np.array([
        [0],
        [0],
        [np.random.normal(0, 0.05)],
        [0]
    ])
    done = False
    step = 0
    while not done and step < 200:
        action = policy(state)
        state = f(state, np.array([[action]]))
        rewards[episode] += 1
        step += 1
print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))

#%%