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
import utilities

#%% Initialize environment
# import gym
# env = gym.make('env:CartPoleControlEnv-v0')

state_dim = 1 # 3
action_dim = 1
A_shape = [state_dim,state_dim]

A = np.zeros(A_shape)
max_abs_real_eigen_val = 1.0
step = 0
while (max_abs_real_eigen_val >= 1.0 or max_abs_real_eigen_val <= 0.7) and (step < 100000):
    Z = np.random.rand(*A_shape)
    A = Z.T @ Z
    W,V = np.linalg.eig(A)
    max_abs_real_eigen_val = np.max(np.abs(np.real(W)))
# A = np.array([
#     [1.0, 0.02,  0.0,        0.0 ],
#     [0.0, 1.0,  -0.01434146, 0.0 ],
#     [0.0, 0.0,   1.0,        0.02],
#     [0.0, 0.0,   0.3155122,  1.0 ]
# ])
W,V = np.linalg.eig(A)
max_abs_real_eigen_val = np.max(np.abs(np.real(W)))
print("A:", A)
print("A's max absolute real eigenvalue:", max_abs_real_eigen_val)

B = np.ones([A_shape[0],action_dim])
# B = np.array([
#     [0],
#     [0.0195122],
#     [0],
#     [-0.02926829]
# ])

def f(x, u):
    return A @ x + B @ u

#%% Define cost
# Q_ = np.eye(A.shape[1])
# R = 1

Q_ = np.eye(A.shape[1])
R = 1 # 0.001

# Q_ = np.array([
#     [10.0, 0.0,  0.0, 0.0],
#     [ 0.0, 1.0,  0.0, 0.0],
#     [ 0.0, 0.0, 10.0, 0.0],
#     [ 0.0, 0.0,  0.0, 1.0]
# ])
# R = 0.1

w_r = np.array([
    [10.0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    # x.T Q x + u.T R u
    x_ = x - w_r
    mat = np.vstack(np.diag(x_.T @ Q_ @ x_)) + np.power(u, 2)*R
    return mat.T

#%% Initialize important vars
state_range = 50
action_range = 50
state_order = 2
action_order = 2
state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

step_size = 0.1
all_us = np.arange(-action_range, action_range+step_size, step_size)
all_us = np.round(all_us, decimals=2)

gamma = 0.99 # this gamma value gets incremented until 0.99
print(f"gamma: {gamma}")
lamb = 1 # 0.0001

#%% Construct datasets
num_episodes = 200
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Path-based approach
# X = np.zeros([state_dim,N])
# U = np.zeros([action_dim,N])
# Y = np.zeros([state_dim,N])
# for episode in range(num_episodes):
#     state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
#     done = False
#     step = 0
#     for step in range(num_steps_per_episode):
#         X[:,(episode*num_steps_per_episode)+step] = state[:,0]
#         # u = np.random.choice(all_us)
#         u = optimal_policy(state)[0,0]
#         U[0,(episode*num_steps_per_episode)+step] = u
#         Y[:,(episode*num_steps_per_episode)+step] = f(state, np.array([[ u ]]))[:, 0]
#         state = np.vstack(Y[:,(episode*num_steps_per_episode)+step])

# Shotgun-based approach
X = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
U = np.random.rand(B.shape[1],N)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],N))
# U = optimal_policy(X)
Y = f(X, U)

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Shotgun-based training error
training_norms = np.zeros([X.shape[1]])
state_norms = np.zeros([X.shape[1]])
for i in range(X.shape[1]):
    state = np.vstack(X[:,i])
    state_norms[i] = utilities.l2_norm(state, np.zeros_like(state))
    action = np.vstack(U[:,i])
    true_x_prime = np.vstack(Y[:,i])
    predicted_x_prime = tensor.f(state, action)
    training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
average_training_norm = np.mean(training_norms)
average_state_norm = np.mean(state_norms)
print(f"Average training norm: {average_training_norm}")
print(f"Average training norm normalized by average state norm: {average_training_norm / average_state_norm}")

#%% Path-based training error
# training_norms = np.zeros([num_episodes,num_steps_per_episode])
# state_norms = np.zeros([X.shape[1]])
# for episode in range(num_episodes):
#     for step in range(num_steps_per_episode):
#         state = np.vstack(X[:,(episode*num_steps_per_episode)+step])
#         state_norms[(episode*num_steps_per_episode)+step] = utilities.l2_norm(state, np.zeros_like(state))
#         action = np.vstack(U[:,(episode*num_steps_per_episode)+step])
#         true_x_prime = np.vstack(Y[:,(episode*num_steps_per_episode)+step])
#         predicted_x_prime = tensor.f(state, action)
#         training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
#         state = true_x_prime
# average_training_norm_per_episode = np.mean(np.sum(training_norms, axis=1))
# average_state_norm = np.mean(state_norms)
# print(f"Average training norm per episode over {num_episodes} episodes: {average_training_norm_per_episode}")
# print(f"Average training norm per episode over {num_episodes} episodes normalized by average state norm: {average_training_norm_per_episode / average_state_norm}")

#%% Pytorch setup
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, 1)
)
model.apply(init_weights)

# model = torch.load('lqr-value-model.pt')

learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

epochs = 500
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
        if (epoch+1) % 250 == 0:
            # np.save('bellman_errors.npy', bellman_errors)
            torch.save(model, 'lqr-value-model.pt')
            print(f"Bellman error at epoch {epoch+1}: {BE}")

        if BE <= epsilon:
            torch.save(model, 'lqr-value-model.pt')
            break

    gamma += 0.01

#%% Optimal policy
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q_, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)
def optimal_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Extract latest policy
def learned_policy(x):
    with torch.no_grad():
        pis_response = pis(x)[:,0]
    return np.random.choice(all_us, p=pis_response.data.numpy())

#%% Test
def watch_agent(num_episodes, num_steps_per_episode):
    optimal_states = np.zeros([num_episodes,num_steps_per_episode,state_dim])
    learned_states = np.zeros([num_episodes,num_steps_per_episode,state_dim])
    optimal_costs = np.zeros([num_episodes])
    learned_costs = np.zeros([num_episodes])

    for episode in range(num_episodes):
        state = np.random.rand(state_dim,1)*state_range*np.random.choice(np.array([-1,1]), size=(state_dim,1))
        optimal_state = state
        learned_state = state
        for step in range(num_steps_per_episode):
            optimal_states[episode,step] = optimal_state[:,0]
            learned_states[episode,step] = learned_state[:,0]

            optimal_action = optimal_policy(optimal_state)
            optimal_state = f(optimal_state, optimal_action)
            optimal_costs[episode] += cost(optimal_state, optimal_action)

            learned_u = learned_policy(learned_state)
            learned_action = np.array([[learned_u]])
            learned_state = f(learned_state, learned_action)
            learned_costs[episode] += cost(learned_state, learned_action)

    print("Norm between entire path (final episode):", utilities.l2_norm(optimal_states[-1], learned_states[-1]))
    print(f"Average cost per episode (optimal controller): {np.mean(optimal_costs)}")
    print(f"Average cost per episode (learned controller): {np.mean(learned_costs)}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('True dynamics')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Learned dynamics')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    # labels = np.array(['cart position', 'cart velocity', 'pole angle', 'pole angular velocity'])
    labels = np.array(['x_0', 'x_1', 'x_2', 'x_3'])
    for i in range(A.shape[1]):
        axs[0].plot(optimal_states[-1,:,i], label=labels[i])
        axs[1].plot(learned_states[-1,:,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

watch_agent(num_episodes=100, num_steps_per_episode=200)

#%% Test model in gym environment
# num_episodes = 100
# num_steps_per_episode = 200
# rewards = np.zeros([num_episodes])
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     for steps in range(num_steps_per_episode):
#         # env.render()
#         action = optimal_policy(np.vstack(state))
#         # action = np.array([[ learned_policy(np.vstack(state)) ]])
#         state, _, done, ___ = env.step(action[0])
#         rewards[episode] += 1
#         if done:
#             break
# # env.close()
# print("Average reward per episode:", np.mean(rewards))

#%%
