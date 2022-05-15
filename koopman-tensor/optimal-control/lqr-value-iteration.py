#%% Imports
import numpy as np
import random
import torch

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

from control import dare
from matplotlib import pyplot as plt
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables

#%% Initialize environment
A_shape = [2,2]
A = np.zeros(A_shape)
max_real_eigen_val = 1.0
# while max_real_eigen_val >= 1.0 or max_real_eigen_val <= 0.7:
# while max_real_eigen_val >= 1.1 or max_real_eigen_val <= 1.0: # SVD does not converge!
while max_real_eigen_val >= 0.5:
    Z = np.random.rand(*A_shape)
    A = Z.T @ Z
    W,V = np.linalg.eig(A)
    max_real_eigen_val = np.max(np.real(W))
print("Max real eigenvalue:", max_real_eigen_val)
print("A:", A)
# B = np.array([
#     [0.0],
#     [0.02],
#     [0.0],
#     [-0.03]
# ])
B = np.ones([A_shape[0],1])

def f(x, u):
    return A @ x + B @ u

#%% Initialize important vars
state_range = 20
action_range = 20
state_order = 2
action_order = 2
state_dim = A_shape[1]
action_dim = B.shape[1]
state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
phi_dim = int(comb( state_order+state_dim, state_order ))
psi_dim = int(comb( action_order+action_dim, action_order ))

step_size = 0.1
all_us = np.arange(-action_range, action_range+step_size, step_size)
all_us = np.round(all_us, decimals=2)

#%% Construct snapshots of u from random agent and initial states x0
N = 200000 # Number of datapoints
X = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])
Y = np.zeros([state_dim,N])
i = 0
while i < N:
    state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
    done = False
    step = 0
    for step in range(1000):
        X[:,i] = state[:,0]
        U[0,i] = np.random.choice(all_us)
        Y[:,i] = f(state, np.array([[ U[0,i] ]]))[:, 0]
        state = np.vstack(Y[:,i])
        i += 1

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
# training_norms = np.zeros([N])
# for step in range(N):
#     x = np.vstack(X[:,step])
#     u = np.vstack(U[:,step])

#     true_x_prime = np.vstack(Y[:,step])
#     predicted_x_prime = tensor.f(x, u)

#     training_norms[step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
# print(f"Mean training norm per episode over {N} steps:", np.mean( training_norms ))

#%% Testing error
# num_episodes = 200
# testing_norms = np.zeros([num_episodes])
# for episode in range(num_episodes):
#     state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
#     done = False
#     step = 0
#     while step < 200:
#         action = np.random.choice(all_us, size=action_column_shape)
#         true_x_prime = f(state, action)
#         predicted_x_prime = tensor.f(state, action)

#         testing_norms[episode] += utilities.l2_norm(true_x_prime, predicted_x_prime)

#         x = true_x_prime
#         step += 1
# print(f"Mean testing norm per episode over {num_episodes} episodes:", np.mean( testing_norms ))

#%% Pytorch setup
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

# model = torch.nn.Sequential(
#     torch.nn.Linear(phi_dim, 1)
# )
# model.apply(init_weights)

model = torch.load('lqr-value-iteration.pt')

gamma = 0.99
lamb = 0.0001
learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% Define cost
# def cost(x, u):
#     return -cartpole_reward.defaultCartpoleRewardMatrix(x, u)

# Q_ = np.array([
#     [10.0, 0.0,  0.0, 0.0],
#     [ 0.0, 1.0,  0.0, 0.0],
#     [ 0.0, 0.0, 10.0, 0.0],
#     [ 0.0, 0.0,  0.0, 1.0]
# ])
# R = 0.1
Q_ = np.eye(A.shape[0])
R = 1.0
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
    delta = np.finfo(np.float32).eps # 1e-25

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

epochs = 0 # 1000
epsilon = 0.00126 # 1e-5
batch_size = 2**9
batch_scale = 3
bellman_errors = [discrete_bellman_error(batch_size*batch_scale).data.numpy()]
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
    BE = discrete_bellman_error(batch_size*batch_scale).data.numpy()
    bellman_errors = np.append(bellman_errors, BE)

    # Every so often, print out and save the bellman error(s)
    if count % 500 == 0:
        # np.save('bellman_errors.npy', bellman_errors)
        torch.save(model, 'lqr-value-iteration.pt')
        print("Current Bellman error:", BE)

    count += 1

    if BE <= epsilon:
        break

#%% Optimal policy
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q_, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)
def optimal_policy(x):
    return np.random.normal(-(C @ x), sigma_t)

#%% Extract latest policy
def learned_policy(x):
    with torch.no_grad():
        pis_response = pis(x)[:,0]
    return np.random.choice(all_us, p=pis_response.data.numpy())

#%% Test learned vs optimal policies
num_episodes = 1#00
num_steps_per_episode = 1000
optimal_states = np.zeros([num_episodes,num_steps_per_episode,state_dim])
learned_states = np.zeros([num_episodes,num_steps_per_episode,state_dim])
optimal_costs = np.zeros([num_episodes])
learned_costs = np.zeros([num_episodes])
for episode in range(num_episodes):
    state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
    optimal_state = state
    learned_state = state
    done = False
    step = 0
    for step in range(num_steps_per_episode):
        optimal_states[episode,step] = optimal_state[:, 0]
        optimal_action = optimal_policy(optimal_state)
        optimal_state = f(optimal_state, optimal_action)
        optimal_costs[episode] += cost(optimal_state, optimal_action)

        learned_states[episode,step] = learned_state[:, 0]
        learned_action = learned_policy(learned_state)
        learned_state = f(learned_state, np.array([[learned_action]]))
        learned_costs[episode] += cost(learned_state, np.array([[learned_action]]))
print(f"Mean optimal cost per episode over {num_episodes} episode(s):", np.mean(optimal_costs))
print(f"Mean learned cost per episode over {num_episodes} episode(s):", np.mean(learned_costs))

#%% Plot
for i in range(state_dim):
    plt.plot(optimal_states[-1,:,i])
plt.show()

for i in range(state_dim):
    plt.plot(learned_states[-1,:,i])
plt.show()

#%%