import gym
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
# env_string = 'CartPole-v0'
env_string = 'env:CartPoleControlEnv-v0'
env = gym.make(env_string)

#%% Initialize important vars
order = 2
x_dim = env.observation_space.shape[0]
u_dim = 1
x_column_shape = [x_dim, 1]
u_column_shape = [u_dim, 1]
Ns = np.arange( x_dim - 1, x_dim - 1 + (order+1) )
phi_dim = int( torch.sum( torch.from_numpy( comb( Ns, np.ones_like(Ns) * (order+1) ) ) ) )

step_size = 1
u_bounds = np.array([[0, 1+step_size]]) if env_string == 'CartPole-v0' else np.array([[-5, 5+step_size]])
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1], step=step_size).reshape(1,-1)
All_U = np.round(All_U, decimals=1) # (1, num_actions)

#%% Construct snapshots of u from random agent and initial states x0
N = 20000 # Number of datapoints
U = np.zeros([u_dim,N])
X = np.zeros([x_dim,N])
Y = np.zeros([x_dim,N])
i = 0
while i < N:
    x = env.reset()
    done = False
    while i < N and not done:
        X[:,i] = x
        U[0,i] = np.random.choice(All_U[0])
        action = int(U[0,i]) if env_string == 'CartPole-v0' else U[:,i]
        Y[:,i], _, done, __ = env.step( action )
        x = Y[:,i]
        i += 1

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(order),
    psi=observables.monomials(2),
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
    x = env.reset()
    done = False
    while not done:
        u = np.random.choice(All_U[0])
        action = int(u) if env_string == 'CartPole-v0' else np.array([u])

        true_x_prime, _, done, ___ = env.step(action)
        predicted_x_prime = tensor.f(
            np.vstack(x),
            action
        )[:,0]

        testing_norms[episode] += utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
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

    V_x_primes_arr = torch.zeros([All_U.shape[1], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

    inner_pi_us_values = -(torch.from_numpy(cost(xs, us)).float() + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = 1e-25

    inner_pi_us_response = torch.real(inner_pi_us(All_U, xs)) # All_U.shape[1] x xs.shape[1]

    # Max trick
    max_inner_pi_u = torch.amax(inner_pi_us_response, axis=0) # xs.shape[1]
    diff = inner_pi_us_response - max_inner_pi_u

    pi_us = torch.exp(diff) + delta # All_U.shape[1] x xs.shape[1]
    Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]
    
    return pi_us / Z_x # All_U.shape[1] x xs.shape[1]

def discrete_bellman_error(batch_size):
    ''' Equation 12 in writeup '''
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # X.shape[0] x batch_size
    phi_xs = tensor.phi(x_batch) # dim_phi x batch_size
    phi_x_primes = tensor.K_(All_U) @ phi_xs # All_U.shape[1] x dim_phi x batch_size

    # 
    pis_response = pis(x_batch) # All_U.shape[1] x x_batch_size
    log_pis = torch.log(pis_response) # All_U.shape[1] x batch_size

    # Compute V(x)'s
    V_x_primes_arr = torch.zeros([All_U.shape[1], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T
    
    # Get costs
    costs = torch.from_numpy(cost(x_batch, All_U)).float() # All_U.shape[1] x batch_size

    # Compute expectations
    expectation_us = (costs + lamb*log_pis + gamma*V_x_primes_arr) * pis_response # All_U.shape[1] x batch_size
    expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

    # Use model to get V(x) for all phi(x)s
    V_xs = model(torch.from_numpy(phi_xs.T).float()) # (batch_size, 1)

    # Compute squared differences
    squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
    total = torch.sum(squared_differences) / batch_size # scalar

    return total

epochs = 1000000
epsilon = 0.00126 # 1e-5
batch_size = 2**10 # 2**9 # 512
bellman_errors = [discrete_bellman_error(batch_size*3).data.numpy()] #* 3 is randomly chosen
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
    pis_response = pis(x_batch) # (All_U.shape[1], batch_size)
    log_pis = torch.log(pis_response) # (All_U.shape[1], batch_size)

    # Compute V(x)'
    phi_x_primes = tensor.K_(All_U) @ phi_x_batch # All_U.shape[1] x dim_phi x batch_size
    V_x_primes_arr = torch.zeros([All_U.shape[1], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T

    # Get costs
    costs = torch.from_numpy(cost(x_batch, All_U)).float() # (All_U.shape[1], batch_size)

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
    BE = discrete_bellman_error(batch_size*3).data.numpy()
    bellman_errors = np.append(bellman_errors, BE)

    # Every so often, print out and save the bellman error(s)
    if count % 500 == 0:
        np.save('bellman_errors.npy', bellman_errors)
        print("Current Bellman error:", BE)

    count += 1

    if BE <= epsilon:
        break

#%% Extract latest policy
All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis_response = pis(x)[:,0]
    u_ind = np.random.choice(All_U_range, p=pis_response.data.numpy())
    u = np.vstack(All_U[:,u_ind])
    return u[0,0]

#%% Test learned policy
num_episodes = 1000
rewards = np.zeros([num_episodes])
for episode in range(num_episodes):
    state = np.vstack(env.reset())
    done = False
    while not done:
        # env.render()
        u = policy(state)
        action = u if env_string == 'CartPole-v0' else np.array([u])
        state, _, done, __ = env.step(action)
        state = np.vstack(state)
        rewards[episode] += 1
# env.close()
print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))

#%%