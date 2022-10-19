# Imports
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.lqr import LQRPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy

# Set seed
seed = 123
np.random.seed(seed)

# Variables
state_dim = 3
action_dim = 1

# state_range = 25.0
state_range = 5.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

action_range = 75.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

step_size = 0.1
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

gamma = 0.99
reg_lambda = 1.0

state_order = 2
action_order = 2

# Dynamics
max_eigen_factor = np.random.uniform(0.7, 1)
print(f"max eigen factor: {max_eigen_factor}")
Z = np.random.rand(state_dim, state_dim)
_, sigma, _ = np.linalg.svd(Z)
Z = Z * np.sqrt(max_eigen_factor) / np.max(sigma)
A = Z.T @ Z
W, _ = np.linalg.eig(A)
max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

print(f"A: {A}")
print(f"A's max absolute real eigenvalue: {max_abs_real_eigen_val}")

B = np.ones([state_dim, action_dim])

def f(x, u):
    return A @ x + B @ u

# Define cost/reward
Q = np.eye(state_dim)
R = 1

reference_point = np.array([
    [0.0],
    [0.0],
    [0.0]
])

def cost(x, u):
    """
        Assuming that data matrices are passed in for X and U. Columns vectors are snapshots.
    """

    # x.T Q x + u.T R u

    _x = x - reference_point

    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    return mat.T

def reward(x, u):
    return -cost(x, u)

# Construct datasets
num_episodes = 100
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, size=[state_dim, N])
U = np.random.uniform(action_minimums, action_maximums, size=[action_dim, N])
Y = f(X, U)

# Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

# LQR Policy
lqr_policy = LQRPolicy(
    A,
    B,
    Q,
    R,
    reference_point,
    gamma,
    reg_lambda,
    seed=seed
)

# Koopman value iteration policy
koopman_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    'saved_models/lqr-discrete-value-iteration-policy.pt',
    seed=seed
)

# Train Koopman policy
koopman_policy.train(training_epochs=500, batch_size=2**10)

# Test policies
def watch_agent(num_episodes, step_limit, specifiedEpisode=-1):
    lqr_states = np.zeros([num_episodes,step_limit,state_dim])
    lqr_actions = np.zeros([num_episodes,step_limit,action_dim])
    lqr_costs = np.zeros([num_episodes])

    koopman_states = np.zeros([num_episodes,step_limit,state_dim])
    koopman_actions = np.zeros([num_episodes,step_limit,action_dim])
    koopman_costs = np.zeros([num_episodes])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    ).T

    for episode in range(num_episodes):
        state = np.vstack(initial_states[episode])

        lqr_state = state
        koopman_state = state

        lqr_cumulative_cost = 0
        koopman_cumulative_cost = 0

        for step in range(step_limit):
            lqr_states[episode,step] = lqr_state[:,0]
            koopman_states[episode,step] = koopman_state[:,0]

            lqr_action = lqr_policy.get_action(lqr_state)
            # if lqr_action[0,0] > action_range:
            #     lqr_action = np.array([[action_range]])
            # elif lqr_action[0,0] < -action_range:
            #     lqr_action = np.array([[-action_range]])
            lqr_actions[episode,step] = lqr_action

            koopman_action = koopman_policy.get_action(koopman_state)
            koopman_actions[episode,step] = koopman_action

            lqr_cumulative_cost += cost(lqr_state, lqr_action)[0,0]
            koopman_cumulative_cost += cost(koopman_state, lqr_action)[0,0]

            lqr_state = f(lqr_state, lqr_action)
            koopman_state = f(koopman_state, koopman_action)

        lqr_costs[episode] = lqr_cumulative_cost
        koopman_costs[episode] = koopman_cumulative_cost

    print(f"Mean cost per episode over {num_episodes} episode(s) (LQR controller): {np.mean(lqr_costs)}")
    print(f"Mean cost per episode over {num_episodes} episode(s) (Koopman controller): {np.mean(koopman_costs)}\n")

    print(f"Initial state of episode #{specifiedEpisode} (LQR controller): {lqr_states[specifiedEpisode,0]}")
    print(f"Final state of episode #{specifiedEpisode} (LQR controller): {lqr_states[specifiedEpisode,-1]}\n")

    print(f"Initial state of episode #{specifiedEpisode} (Koopman controller): {koopman_states[specifiedEpisode,0]}")
    print(f"Final state of episode #{specifiedEpisode} (Koopman controller): {koopman_states[specifiedEpisode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (LQR controller): {np.abs(lqr_states[specifiedEpisode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (LQR controller): {np.linalg.norm(lqr_states[specifiedEpisode,-1] - reference_point[:,0])}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {np.abs(koopman_states[specifiedEpisode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {np.linalg.norm(koopman_states[specifiedEpisode,-1] - reference_point[:,0])}")

    # Plot dynamics over time for all state dimensions for both controllers
    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')
    axs[0].set_title('LQR Controller')
    axs[0].set(xlabel='Timestep', ylabel='State value')
    axs[1].set_title('Koopman Controller')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        axs[0].plot(lqr_states[specifiedEpisode,:,i], label=labels[i])
        axs[1].plot(koopman_states[specifiedEpisode,:,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.tight_layout()
    plt.show()

    # Plot x_0 vs x_1 for both controller types
    fig, axes = plt.subplots(2)
    fig.suptitle(f"Controllers in Environment (2D; Episode #{specifiedEpisode})")
    axes[0].set_title(f"LQR Controller")
    axes[0].plot(lqr_states[specifiedEpisode,:,0], lqr_states[specifiedEpisode,:,1])
    axes[1].set_title("Koopman Controller")
    axes[1].plot(koopman_states[specifiedEpisode,:,0], koopman_states[specifiedEpisode,:,1])
    plt.show()

    # Labels that will be used for the next two plots
    labels = ['LQR controller', 'Koopman controller']

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(lqr_actions[specifiedEpisode,:,0])
    plt.hist(koopman_actions[specifiedEpisode,:,0])
    plt.legend(labels)
    plt.show()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(lqr_actions.shape[1]), lqr_actions[specifiedEpisode,:,0], s=5)
    plt.scatter(np.arange(koopman_actions.shape[1]), koopman_actions[specifiedEpisode,:,0], s=5)
    plt.legend(labels)
    plt.show()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=200, specifiedEpisode=42)