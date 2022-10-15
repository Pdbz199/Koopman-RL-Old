#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
np.random.seed(seed)

from control import dare
from generalized.discrete_koopman_actor_critic_policy import DiscreteKoopmanActorCriticPolicy

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
    max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

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
# state_range = 25.0
state_range = 5.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

# action_range = 75.0
action_range = 20.0
# action_range = 10.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

step_size = 0.1
# step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

gamma = 0.99
reg_lambda = 1.0

#%% Default policies
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

#%% Optimal policy
P = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q, R)[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = reg_lambda * np.linalg.inv(R + B.T @ P @ B)

def lqr_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Construct datasets
num_episodes = 500
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, [state_dim,N])
U = np.random.uniform(action_minimums, action_maximums, [action_dim,N])
Y = f(X, U)

# Path-based approach
# X = np.zeros([state_dim,N])
# Y = np.zeros([state_dim,N])
# U = np.zeros([action_dim,N])

# initial_states = np.random.uniform(
#     state_minimums,
#     state_maximums,
#     [state_dim, num_episodes]
# )

# for episode in range(num_episodes):
#     x = np.vstack(initial_states[:,episode])
#     for step in range(num_steps_per_episode):
#         X[:,(episode*num_steps_per_episode)+step] = x[:,0]
#         u = random_policy(x)
#         U[:,(episode*num_steps_per_episode)+step] = u[:,0]
#         x = f(x, u)
#         Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

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
policy = DiscreteKoopmanActorCriticPolicy(
    f,
    gamma,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'lqr-policy-iteration.pt',
    seed=seed,
    # learning_rate=0.0001
)
policy.actor_critic(
    num_training_episodes=10000,
    num_steps_per_episode=200
)

#%% Test
def watch_agent(num_episodes, test_steps):
    # specifiedEpisode = -1
    specifiedEpisode = 42

    lqr_states = np.zeros([num_episodes,state_dim,test_steps])
    lqr_actions = np.zeros([num_episodes,action_dim,test_steps])
    lqr_costs = np.zeros([num_episodes])

    koopman_states = np.zeros([num_episodes,state_dim,test_steps])
    koopman_actions = np.zeros([num_episodes,action_dim,test_steps])
    koopman_costs = np.zeros([num_episodes])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    )

    for episode in range(num_episodes):
        state = np.vstack(initial_states[:,episode])

        lqr_state = state
        koopman_state = state

        for step in range(test_steps):
            lqr_states[episode,:,step] = lqr_state[:,0]
            koopman_states[episode,:,step] = koopman_state[:,0]

            lqr_action = lqr_policy(lqr_state)
            # if lqr_action[0,0] > action_range:
            #     lqr_action = np.array([[action_range]])
            # elif lqr_action[0,0] < -action_range:
            #     lqr_action = np.array([[-action_range]])
            lqr_actions[episode,:,step] = lqr_action[:,0]

            with torch.no_grad():
                koopman_u, _ = policy.get_action(koopman_state[:,0])
            koopman_action = np.array([[koopman_u]])
            koopman_actions[episode,:,step] = koopman_action[:,0]

            lqr_costs[episode] += cost(lqr_state, lqr_action)[0,0]
            koopman_costs[episode] += cost(koopman_state, koopman_action)[0,0]

            lqr_state = f(lqr_state, lqr_action)
            koopman_state = f(koopman_state, koopman_action)

    print(f"Mean cost per episode over {num_episodes} episode(s) (LQR controller): {np.mean(lqr_costs)}")
    print(f"Mean cost per episode over {num_episodes} episode(s) (Koopman controller): {np.mean(koopman_costs)}\n")

    print(f"Initial state of episode #{specifiedEpisode} (LQR controller): {lqr_states[specifiedEpisode,:,0]}")
    print(f"Final state of episode #{specifiedEpisode} (LQR controller): {lqr_states[specifiedEpisode,:,-1]}\n")

    print(f"Initial state of episode #{specifiedEpisode} (Koopman controller): {koopman_states[specifiedEpisode,:,0]}")
    print(f"Final state of episode #{specifiedEpisode} (Koopman controller): {koopman_states[specifiedEpisode,:,-1]}\n")

    print(f"Reference state: {w_r[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (LQR controller): {np.abs(lqr_states[specifiedEpisode,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (LQR controller): {utilities.l2_norm(lqr_states[specifiedEpisode,:,-1], w_r[:,0])}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {np.abs(koopman_states[specifiedEpisode,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {utilities.l2_norm(koopman_states[specifiedEpisode,:,-1], w_r[:,0])}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('LQR Controller')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Koopman Controller')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
    for i in range(state_dim):
        axs[0].plot(lqr_states[specifiedEpisode,i], label=labels[i])
        axs[1].plot(koopman_states[specifiedEpisode,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

    plt.plot(lqr_states[specifiedEpisode,0], lqr_states[specifiedEpisode,1])
    plt.title("LQR Controller in Environment (2D)")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.show()

    plt.plot(koopman_states[specifiedEpisode,0], koopman_states[specifiedEpisode,1])
    plt.title("Koopman Controller in Environment (2D)")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.show()

    labels = ['LQR controller', 'Koopman controller']

    plt.hist(lqr_actions[specifiedEpisode,0])
    plt.hist(koopman_actions[specifiedEpisode,0])
    plt.legend(labels)
    plt.title(f"Histogram Of Actions (In Episode #{specifiedEpisode})")
    plt.show()

    plt.scatter(np.arange(lqr_actions.shape[2]), lqr_actions[specifiedEpisode,0], s=5)
    plt.scatter(np.arange(koopman_actions.shape[2]), koopman_actions[specifiedEpisode,0], s=5)
    plt.legend(labels)
    plt.title(f"Scatter Plot of Actions Per Step (In Episode #{specifiedEpisode})")
    plt.show()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, test_steps=200)