#%% Imports
import numpy as np
import random
import torch

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

from enum import IntEnum

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import utilities

#%% Constants
TWO_THIRDS = 2/3
gamma = 0.99
lamb = 0.0001

#%% IntEnum for high/low, up/down
class State(IntEnum):
    HIGH = 1
    LOW = 0

class Action(IntEnum):
    UP = 1
    DOWN = 0

#%% System dynamics
def f(x, u):
    random = np.random.rand()
    if x == State.HIGH and u == Action.UP:
        return State.HIGH if random <= TWO_THIRDS else State.LOW
    elif x == State.HIGH and u == Action.DOWN:
        return State.LOW if random <= TWO_THIRDS else State.HIGH
    elif x == State.LOW and u == Action.UP:
        return State.HIGH if random <= TWO_THIRDS else State.LOW
    elif x == State.LOW and u == Action.DOWN:
        return State.LOW if random <= TWO_THIRDS else State.HIGH

#%% Define cost
def cost(x, u):
    if x == State.HIGH and u == Action.UP:
        return 1.0
    elif x == State.HIGH and u == Action.DOWN:
        return 100.0
    elif x == State.LOW and u == Action.UP:
        return 100.0
    elif x == State.LOW and u == Action.DOWN:
        return 1.0

def costs(xs, us):
    costs = np.empty((xs.shape[1],us.shape[1]))
    for i in range(xs.shape[1]):
        x = np.vstack(xs[:,i])
        for j in range(us.shape[1]):
            u = np.vstack(us[:,j])
            costs[i,j] = cost(x, u)

    return costs

#%% Construct snapshots of u from random agent and initial states x0
N = 100
X = np.random.choice(list(State), size=(1,N))
U = np.random.choice(list(Action), size=(1,N))

#%% Construct snapshots of states following dynamics f(x,u) -> x'
Y = np.empty_like(X)
for i in range(X.shape[1]):
    x = np.vstack(X[:,i])
    u = np.vstack(U[:,i])
    Y[:,i] = f(x, u)

#%% Dictionaries
distinct_xs = 2
distinct_us = 2

enumerated_states = np.array([State.HIGH, State.LOW])
enumerated_actions = np.array([Action.UP, Action.DOWN])

def phi(x):
    phi_x = np.zeros((distinct_xs,x.shape[1]))
    phi_x[x[0].astype(int),np.arange(0,x.shape[1])] = 1
    return phi_x

def psi(u):
    psi_u = np.zeros((distinct_us,u.shape[1]))
    psi_u[u[0].astype(int),np.arange(0,u.shape[1])] = 1
    return psi_u

#%% Koopman tensor
tensor = KoopmanTensor(X, Y, U, phi, psi)

#%% Training error
norms = np.empty((N))
for i in range(N):
    phi_x = np.vstack(tensor.Phi_X[:,i]) # current (lifted) state

    action = np.vstack(U[:,i])

    true_x_prime = np.vstack(Y[:,i])
    predicted_x_prime = tensor.B.T @ tensor.K_(action) @ phi_x

    # Compute norms
    norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
print("Average training error:", np.mean(norms))

#%% Testing error
num_episodes = 100
num_steps_per_episode = 100

norms = np.zeros((num_episodes))
for episode in range(num_episodes):
    x = np.array([[np.random.choice(list(State))]])

    for step in range(num_steps_per_episode):
        phi_x = phi(x) # apply phi to state

        action = np.array([[np.random.choice(list(Action))]]) # sample random action

        true_x_prime = np.array([[f(x, action)]])
        predicted_x_prime = tensor.B.T @ tensor.K_(action) @ phi_x

        norms[episode] += utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
print("Average testing error per episode:", np.mean(norms))

#%% Pytorch setup
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

# model = torch.nn.Sequential(
#     torch.nn.Linear(distinct_xs, 1)
# )
# model.apply(init_weights)

model = torch.load('simple-mdp-value-iteration.pt')

learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% Control
def inner_pi_us(us, xs):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]

    V_x_primes_arr = torch.zeros([enumerated_actions.shape[0], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

    inner_pi_us_values = -(torch.from_numpy(costs(xs, us).T).float() + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = np.finfo(np.float32).eps # 1e-25

    inner_pi_us_response = torch.real(inner_pi_us(np.array([enumerated_actions]), xs)) # all_us.shape[0] x xs.shape[1]

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
    phi_x_primes = tensor.K_(np.array([enumerated_actions])) @ phi_xs # all_us.shape[0] x dim_phi x batch_size

    pis_response = pis(x_batch) # all_us.shape[0] x x_batch_size
    log_pis = torch.log(pis_response) # all_us.shape[0] x batch_size

    # Compute V(x)'s
    V_x_primes_arr = torch.zeros([enumerated_actions.shape[0], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T
    
    # Get costs
    cost_vals = torch.from_numpy(costs(x_batch, np.array([enumerated_actions])).T).float() # all_us.shape[0] x batch_size

    # Compute expectations
    expectation_us = (cost_vals + lamb*log_pis + gamma*V_x_primes_arr) * pis_response # all_us.shape[0] x batch_size
    expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

    # Use model to get V(x) for all phi(x)s
    V_xs = model(torch.from_numpy(phi_xs.T).float()) # (batch_size, 1)

    # Compute squared differences
    squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
    total = torch.sum(squared_differences) / batch_size # scalar

    return total
    
epochs = 0 # 5000
epsilon = 0.01
batch_size = N
batch_scale = 1
bellman_errors = [discrete_bellman_error(batch_size*batch_scale).data.numpy()]
BE = bellman_errors[-1]
print("Initial Bellman error:", BE)

for epoch in range(epochs):
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
    phi_x_primes = tensor.K_(np.array([enumerated_actions])) @ phi_x_batch # all_us.shape[0] x dim_phi x batch_size
    V_x_primes_arr = torch.zeros([enumerated_actions.shape[0], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T

    # Get costs
    cost_vals = torch.from_numpy(costs(x_batch, np.array([enumerated_actions])).T).float() # (all_us.shape[0], batch_size)

    # Compute expectations
    expectation_term_1 = torch.sum(
        torch.mul(
            (cost_vals + lamb*log_pis + gamma*V_x_primes_arr),
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
    if (epoch+1) % 500 == 0:
        # np.save('bellman_errors.npy', bellman_errors)
        torch.save(model, 'simple-mdp-value-iteration.pt')
        print(f"Bellman error at epoch {epoch+1}: {BE}")

    if BE <= epsilon:
        torch.save(model, 'simple-mdp-value-iteration.pt')
        break

#%% Extract latest policy
def learned_policy(x):
    with torch.no_grad():
        pis_response = pis(x)[:,0]
    print(pis_response)
    return np.random.choice(enumerated_actions, p=pis_response.data.numpy())

#%% Test policy by simulating system
num_episodes = 100
num_steps_per_episode = 100

cost_vals = np.empty((num_episodes))
for episode in range(num_episodes):
    x = np.array([[np.random.choice(list(State))]])

    cost_sum = 0
    for step in range(num_steps_per_episode):
        u = learned_policy(x)

        cost_sum += cost(x, u)

        x = np.array([[f(x, u)]])

    cost_vals[episode] = cost_sum
print("Mean cost per episode:", np.mean(cost_vals)) # Cost should be minimized

#%%