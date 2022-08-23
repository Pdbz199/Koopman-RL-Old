#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import dare
from generalized.discrete_koopman_value_iteration_policy import DiscreteKoopmanValueIterationPolicy

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% Initialize environment
state_dim = 3
action_dim = 1

A = np.zeros([state_dim, state_dim])
max_abs_real_eigen_val = 1.0
while max_abs_real_eigen_val >= 1.0 or max_abs_real_eigen_val <= 0.7:
    Z = np.random.rand(*A.shape)
    _,sigma,__ = np.linalg.svd(Z)
    Z /= np.max(sigma)
    A = Z.T @ Z
    W,_ = np.linalg.eig(A)
    max_eigenvalue = np.max(np.absolute(W, W))
    max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

print(f"Maximum eigenvalue: {max_eigenvalue}")
print("A:", A)
print("A's max absolute real eigenvalue:", max_abs_real_eigen_val)
B = np.ones([state_dim,action_dim])

def f(x, u):
    return A @ x + B @ u

#%% Define cost
Q = np.eye(state_dim)
R = 1
w_r = np.array([
    [0.0],
    [0.0],
    [0.0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns are snapshots
    # x.T Q x + u.T R u
    x_ = x - w_r
    mat = np.vstack(np.diag(x_.T @ Q @ x_)) + np.power(u, 2)*R
    return mat.T

#%% Initialize important vars
state_range = 10.0
state_minimums = np.ones([state_dim,1]) * -state_range + 20
state_maximums = np.ones([state_dim,1]) * state_range

action_range = 25.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

step_size = 0.1
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

gamma = 0.99
reg_lambda = 1.0

#%% Optimal policy
P = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q, R)[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = reg_lambda * np.linalg.inv(R + B.T @ P @ B)

def optimal_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Construct datasets
num_episodes = 100
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, [state_dim,N])
U = np.random.uniform(action_minimums, action_maximums, [action_dim,N])
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

#%% Compute optimal policy
policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'lqr-value-iteration.pt'
)
policy.train(
    training_epochs=500,
    batch_size=2**9,
    batch_scale=3,
    epsilon=1e-2,
    gamma_increment_amount=0.02
)

#%% Test
test_steps = 200
def watch_agent():
    optimal_states = np.zeros([num_episodes,test_steps,state_dim])
    learned_states = np.zeros([num_episodes,test_steps,state_dim])
    optimal_actions = np.zeros([num_episodes,test_steps,action_dim])
    learned_actions = np.zeros([num_episodes,test_steps,action_dim])
    optimal_costs = np.zeros([num_episodes])
    learned_costs = np.zeros([num_episodes])

    for episode in range(num_episodes):
        state = np.random.rand(state_dim,1)*state_range*np.random.choice(np.array([-1,1]), size=(state_dim,1))
        optimal_state = state
        learned_state = state
        step = 0
        while step < test_steps:
            optimal_states[episode,step] = optimal_state[:,0]
            learned_states[episode,step] = learned_state[:,0]

            optimal_action = optimal_policy(optimal_state)
            optimal_state = f(optimal_state, optimal_action)
            optimal_costs[episode] += cost(optimal_state, optimal_action)

            with torch.no_grad():
                learned_action = policy.get_action(learned_state)
            learned_state = f(learned_state, learned_action)
            learned_costs[episode] += cost(learned_state, learned_action)

            optimal_actions[episode,step] = optimal_action[:,0]
            learned_actions[episode,step] = learned_action[:,0]

            step += 1

    print("Norm between entire path (final episode):", utilities.l2_norm(optimal_states[-1], learned_states[-1]))
    print(f"Average cost per episode (optimal controller): {np.mean(optimal_costs)}")
    print(f"Average cost per episode (learned controller): {np.mean(learned_costs)}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('LQR Controller')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Koopman Controller')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
    for i in range(A.shape[1]):
        axs[0].plot(optimal_states[-1,:,i], label=labels[i])
        axs[1].plot(learned_states[-1,:,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

    plt.hist(learned_actions[-1,:,0])
    plt.show()

    plt.scatter(np.arange(learned_actions.shape[1]), learned_actions[-1,:,0], s=5)
    plt.show()

watch_agent()

#%%