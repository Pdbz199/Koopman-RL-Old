#%% Imports
import gym
env = gym.make('env:CartPoleControlEnv-v0')
import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
np.random.seed(123)
torch.manual_seed(seed)

from control import dlqr
from scipy.special import comb

import sys
sys.path.append('../../')
from tensor import KoopmanTensor
sys.path.append('../../../')
import observables

#%% System dynamics
# Cartpole A, B, Q, and R matrices from Wen's homework
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

#%% Cost
Q_ = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1

#%% Traditional LQR solution
C = dlqr(A, B, Q_, R)[0]

#%% Construct snapshots of data
action_range = 25
state_range = 50

step_size = 0.1
all_us = np.arange(-action_range, action_range+step_size, step_size)
# all_us = np.arange(-10, 10+step_size, step_size)

num_episodes = 300
num_steps_per_episode = 500
N = num_episodes * num_steps_per_episode

# Shotgun approach
X = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
U = np.random.rand(B.shape[1],N)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],N))
Y = f(X, U)

#%% Estimate Koopman tensor
state_order = 2
action_order = 2

phi_dim = int(comb( state_order+A.shape[1], state_order ))
psi_dim = int(comb( action_order+B.shape[1], action_order ))

tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Shotgun-based training error
# training_norms = np.zeros([X.shape[1]])
# state_norms = np.zeros([X.shape[1]])
# for i in range(X.shape[1]):
#     state = np.vstack(X[:,i])
#     state_norms[i] = utilities.l2_norm(state, np.zeros_like(state))
#     action = np.vstack(U[:,i])
#     true_x_prime = np.vstack(Y[:,i])
#     predicted_x_prime = tensor.f(state, action)
#     training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
# average_training_norm = np.mean(training_norms)
# average_state_norm = np.mean(state_norms)
# print(f"Average training norm: {average_training_norm}")
# print(f"Average training norm normalized by average state norm: {average_training_norm / average_state_norm}")

#%% Plot state path
# num_steps = 200
# true_states = np.zeros([num_steps, A.shape[1]])
# true_actions = np.zeros([num_steps, B.shape[1]])
# koopman_states = np.zeros([num_steps, A.shape[1]])
# koopman_actions = np.zeros([num_steps, B.shape[1]])
# state = np.vstack(env.reset())
# true_state = state
# koopman_state = state
# for i in range(num_steps):
#     true_states[i] = true_state[:,0]
#     koopman_states[i] = koopman_state[:,0]
#     true_action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1]), size=(1,1))
#     # true_action = np.random.choice(all_us, size=[B.shape[1],1])
#     # true_action = -(C @ true_state)
#     # koopman_action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1]), size=(1,1))
#     # koopman_action = np.random.choice(all_us, size=[B.shape[1],1])
#     # koopman_action = -(C @ koopman_state)
#     true_state = f(true_state, true_action)
#     koopman_state = tensor.f(koopman_state, true_action)
# print("Norm between entire paths:", utilities.l2_norm(true_states, koopman_states))

# fig, axs = plt.subplots(2)
# fig.suptitle('Dynamics Over Time')

# axs[0].set_title('True dynamics')
# axs[0].set(xlabel='Timestep', ylabel='State value')

# axs[1].set_title('Learned dynamics')
# axs[1].set(xlabel='Timestep', ylabel='State value')

# labels = np.array(['cart position', 'cart velocity', 'pole angle', 'pole angular velocity'])
# for i in range(A.shape[1]):
#     axs[0].plot(true_states[:,i], label=labels[i])
#     axs[1].plot(koopman_states[:,i], label=labels[i])
# lines_labels = [axs[0].get_legend_handles_labels()]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)

# plt.tight_layout()
# plt.show()

#%% PyTorch setup
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, 1)
)
model.apply(init_weights)

# model = torch.load('wen-homework-value-iteration.pt')

#%% Hyperparams
gamma = 0.99
lamb = 0.0001
learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% Define cost
# def cost(x, u):
#     return -cartpole_reward.defaultCartpoleRewardMatrix(x, u)

def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    # x.T Q x + u.T R u
    mat = np.vstack(np.diag(x.T @ Q_ @ x)) + np.power(u, 2)*R
    return mat.T

#%% Control
def inner_pi_us(us, xs):
    K_us = tensor.K_(us) # (us.shape[1], dim_phi, dim_phi)
    phi_xs = tensor.phi(xs) # (dim_phi, xs.shape[1])

    # phi_x_primes = K_us @ phi_xs # us.shape[1] x dim_phi x xs.shape[1]
    phi_x_primes = np.einsum('upq,px->upx', K_us, phi_xs) # (us.shape[1] x dim_phi x xs.shape[1])

    V_x_primes_arr = torch.zeros([all_us.shape[0], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

    inner_pi_us_values = -(torch.from_numpy(cost(xs, us)).float() + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = np.finfo(np.float32).eps # used to be 1e-25

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

epochs = 0 # 50000
epsilon = 0.01
batch_size = 2**14
scale_factor = 1
bellman_errors = [discrete_bellman_error(batch_size*scale_factor).data.numpy()]
BE = bellman_errors[-1]
print("Initial Bellman error:", BE)

# count = 0
# while BE > epsilon:
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
    # phi_x_primes = tensor.K_(np.array([all_us])) @ phi_x_batch # all_us.shape[0] x dim_phi x batch_size
    phi_x_primes = np.einsum('upq,px->upx', tensor.K_(np.array([all_us])), phi_x_batch) # (all_us.shape[0], dim_phi, batch_size)
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
    BE = discrete_bellman_error(batch_size*scale_factor).data.numpy()
    bellman_errors = np.append(bellman_errors, BE)

    # Every so often, print out and save the bellman error(s)
    if (epoch+1) % 100 == 0:
        np.save('bellman_errors.npy', bellman_errors)
        torch.save(model, 'wen-homework-value-iteration.pt')
        print(f"Current Bellman error (epoch {epoch+1}): {BE}")

    # count += 1

    if BE <= epsilon:
        np.save('bellman_errors.npy', bellman_errors)
        torch.save(model, 'wen-homework-value-iteration.pt')
        break

#%% Extract latest policy
def policy(x):
    pis_response = pis(x)[:,0]
    return np.random.choice(all_us, p=pis_response.data.numpy())

#%% Test learned policy
num_episodes = 100
rewards = np.zeros([num_episodes])
for episode in range(num_episodes):
    state = np.vstack(env.reset())
    done = False
    step = 0
    while not done and step < 200:
        action = policy(state)
        state = f(state, np.array([[action]]))
        done = is_done(state)
        rewards[episode] += 1
        step += 1
print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))

#%%