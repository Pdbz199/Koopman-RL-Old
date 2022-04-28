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
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import cartpole_reward
import observables

#%% Initialize environment
env = gym.make('CartPole-v0')

#%% Construct snapshots of u from random agent and initial states x0
N = 20000 # Number of datapoints
U = np.zeros([1,N])
X = np.zeros([4,N+1])
Y = np.zeros([4,N])
i = 0
while i < N:
    X[:,i] = env.reset()
    done = False
    while i < N and not done:
        U[0,i] = env.action_space.sample()
        Y[:,i], _, done, __ = env.step(int(U[0,i]))
        if not done:
            X[:,i+1] = Y[:,i]
        i += 1
X = X[:,:-1]

#%% Estimate Koopman tensor
order = 2
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(order),
    psi=observables.monomials(order),
    regressor='ols'
)

# obs_size = env.observation_space.shape[0]
state_dim = env.observation_space.shape[0]
Ns = np.arange( state_dim - 1, state_dim - 1 + (order+1) )
obs_size = int( np.sum( comb( Ns, np.ones_like(Ns) * (order+1) ) ) )
n_actions = env.action_space.n
# HIDDEN_SIZE = 256

model = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 1)
)
# print("Model:", model)
# input = torch.from_numpy(tensor.phi(np.vstack(env.reset()))[:,0]).float()
# print("Input:", input)
# print("Output:", model(input))

gamma = 0.99
lamb = 0.0001
learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

u_bounds = np.array([[0, 2]])
step_size = 1
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1], step=step_size).reshape(1,-1)
All_U = np.round(All_U, decimals=1) # (1, 2)

def cost(x, u):
    return -cartpole_reward.defaultCartpoleRewardMatrix(x, u)

def inner_pi_us(us, xs):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]
    V_x_primes_arr = np.zeros([All_U.shape[1], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes = model(torch.from_numpy(phi_x_primes[u].T).float()).T # 1 x xs.shape[1]
        V_x_primes_arr[u] = V_x_primes.data.numpy()
    inner_pi_us_values = -(cost(xs, us) + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]
    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = 1e-25
    inner_pi_us_response = np.real(inner_pi_us(All_U, xs)) # All_U.shape[1] x xs.shape[1]
    max_inner_pi_u = np.amax(inner_pi_us_response, axis=0) # xs.shape[1]
    diff = inner_pi_us_response - max_inner_pi_u
    pi_us = np.exp(diff) + delta # All_U.shape[1] x xs.shape[1]
    Z_x = np.sum(pi_us, axis=0) # xs.shape[1]
    
    return pi_us / Z_x # All_U.shape[1] x xs.shape[1]

def discrete_bellman_error(batch_size):
    ''' Equation 12 in writeup '''
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # X.shape[0] x batch_size
    phi_xs = tensor.phi(x_batch) # dim_phi x batch_size
    phi_x_primes = tensor.K_(All_U) @ phi_xs # All_U.shape[1] x dim_phi x batch_size
    pis_response = pis(x_batch) # All_U.shape[1] x x_batch_size
    log_pis = np.log(pis_response) # All_U.shape[1] x batch_size

    # pi_sum = np.sum(pis_response)
    # assert np.isclose(pi_sum, 1, rtol=1e-3, atol=1e-4)

    V_x_primes_arr = np.zeros([All_U.shape[1], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes = model(torch.from_numpy(phi_x_primes[u].T).float()).T
        V_x_primes_arr[u] = V_x_primes.data.numpy()
    costs = cost(x_batch, All_U) # All_U.shape[1] x batch_size
    expectation_us = (costs + lamb*log_pis + gamma*V_x_primes_arr) * pis_response # All_U.shape[1] x batch_size
    expectation_u = np.sum(expectation_us, axis=0) # batch_size

    V_xs = model(torch.from_numpy(phi_xs.T).float()) # batch_size x 1 (1536, 1)
    squared_differences = np.power(V_xs[:,0].data.numpy() - expectation_u, 2) # 1 x batch_size
    total = np.sum(squared_differences) / batch_size # scalar

    return total

epochs = 7500
epsilon = 1e-2
batch_size = 512
bellman_errors = [discrete_bellman_error(batch_size*3)] #* 3 is randomly chosen
BE = bellman_errors[-1]
print("Initial Bellman error:", BE)

count = 0
# while BE > epsilon:
for _ in range(epochs):
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:,x_batch_indices] # X.shape[0] x batch_size
    phi_x_batch = tensor.phi(x_batch) # dim_phi x batch_size

    V_x = model(torch.from_numpy(phi_x_batch.T).float()).T # 1 x batch_size (1, 512)

    pis_response = pis(x_batch) # All_U.shape[1] x batch_size (2, 512)
    log_pis = np.log(pis_response) # All_U.shape[1] x batch_size (2, 512)
    phi_x_primes = tensor.K_(All_U) @ phi_x_batch # All_U.shape[1] x dim_phi x batch_size
    V_x_primes_arr = np.zeros([All_U.shape[1], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes = model(torch.from_numpy(phi_x_primes[u].T).float()).T
        V_x_primes_arr[u] = V_x_primes.data.numpy()
    costs = cost(x_batch, All_U)

    expectation_term_1 = np.sum((costs + lamb*log_pis + gamma*V_x_primes_arr) * pis_response, axis=0) # batch_size
    expectation_term_2 = np.einsum('ux,upx->px', pis_response, gamma*phi_x_primes) # dim_phi x batch_size

    # Equations 22/23 in writeup
    difference = torch.sum(
        (V_x - torch.tensor(expectation_term_1)) * \
        (torch.tensor(phi_x_batch) - torch.tensor(expectation_term_2))
    )
    
    optimizer.zero_grad()
    difference.backward()
    optimizer.step()

    # Recompute Bellman error
    BE = discrete_bellman_error(batch_size*3)
    bellman_errors = np.append(bellman_errors, BE)

    if count % 500 == 0:
        np.save('bellman_errors.npy', bellman_errors)
        print("Current Bellman error:", BE)

    count += 1

All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis_response = pis(x)[:,0]
    u_ind = np.random.choice(All_U_range, p=pis_response)
    u = np.vstack(All_U[:,u_ind])
    return u[0,0]

num_episodes = 1000
def watch_agent():
    rewards = np.zeros([num_episodes])
    for episode in range(num_episodes):
        state = np.vstack(env.reset())
        done = False
        episode_rewards = []
        while not done:
            # env.render()
            action = policy(state)
            state, reward, done, _ = env.step(action)
            state = np.vstack(state)
            episode_rewards.append(reward)
        rewards[episode] = np.sum(episode_rewards)
    # env.close()
    print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))
watch_agent()