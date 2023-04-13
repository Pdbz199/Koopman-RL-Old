#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from cost import cost, reference_point
from dynamics import (
    action_dim,
    all_actions,
    dt,
    f,
    state_dim,
    state_maximums,
    state_minimums
)

sys.path.append('../../../')
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy

#%% Load Koopman tensor with pickle
with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

#%% Load LQR policy
with open('./analysis/tmp/lqr/policy.pickle', 'rb') as handle:
    lqr_policy = pickle.load(handle)

#%% Variables
gamma = 0.99
reg_lambda = 1.0

#%% Koopman value iteration policy
koopman_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    save_data_path="./analysis/tmp/discrete_value_iteration",
    seed=seed,
    dt=dt,
    load_model=True
)

#%% Functions for showing/saving figures
def show_plot():
    plt.tight_layout()
    plt.show()

path = "./analysis/tmp/discrete_value_iteration/plots"
plot_file_extensions = ['png', 'svg']
def save_figure(title: str):
    plt.tight_layout()
    for plot_file_extension in plot_file_extensions:
        plt.savefig(f"{path}/{title}.{plot_file_extension}")
    plt.clf()

#%% Test policies
def watch_agent(num_episodes, num_steps_per_episode, specified_episode):
    if specified_episode is None:
        specified_episode = num_episodes-1

    lqr_states = np.zeros((num_episodes, num_steps_per_episode, state_dim))
    lqr_actions = np.zeros((num_episodes, num_steps_per_episode, action_dim))
    lqr_costs = np.zeros((num_episodes, num_steps_per_episode))

    value_iteration_states = np.zeros_like(lqr_states)
    value_iteration_actions = np.zeros_like(lqr_actions)
    value_iteration_costs = np.zeros_like(lqr_costs)

    # Starting randomly in the space
    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, num_episodes]
    ).T

    for episode_num in range(num_episodes):
        # Extract initial state
        state = np.vstack(initial_states[episode_num])

        # Set initial states to be equal
        lqr_state = state
        value_iteration_state = state

        for step_num in range(num_steps_per_episode):
            # Save latest states
            lqr_states[episode_num,step_num] = lqr_state[:, 0]
            value_iteration_states[episode_num,step_num] = value_iteration_state[:, 0]

            # Get actions for current state and save them
            lqr_action = lqr_policy.get_action(lqr_state, is_entropy_regularized=True)
            lqr_actions[episode_num, step_num] = lqr_action[:, 0]
            value_iteration_action = koopman_policy.get_action(value_iteration_state, is_greedy=False)
            value_iteration_actions[episode_num, step_num] = value_iteration_action[:, 0]

            # Compute costs
            lqr_costs[episode_num,step_num] = cost(lqr_state, lqr_action)[0, 0]
            value_iteration_costs[episode_num,step_num] = cost(value_iteration_state, value_iteration_action)[0, 0]

            # Compute next states
            lqr_state = f(lqr_state, lqr_action)
            value_iteration_state = f(value_iteration_state, value_iteration_action)

    # Make figure for plots
    fig = plt.figure(figsize=(16,9))
    episode_range = np.arange(num_episodes)
    step_range = np.arange(num_steps_per_episode)

    # Plot total costs per episode
    ax = fig.add_subplot(3, 3, 1)
    ax.set_title("Total Cost Per Episode")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Total Cost")
    lqr_costs_per_episode = lqr_costs.sum(axis=1)
    value_iteration_costs_per_episode = value_iteration_costs.sum(axis=1)
    ax.plot(lqr_costs_per_episode)
    ax.plot(value_iteration_costs_per_episode)
    ax.scatter(episode_range, lqr_costs_per_episode)
    ax.scatter(episode_range, value_iteration_costs_per_episode)

    print(f"Mean of total LQR costs per episode over {num_episodes} episode(s): {lqr_costs_per_episode.mean()}")
    print(f"Standard deviation of total LQR costs per episode over {num_episodes} episode(s): {lqr_costs_per_episode.std()}")
    print(f"Mean of total value iteration costs per episode over {num_episodes} episode(s): {value_iteration_costs_per_episode.mean()}")
    print(f"Standard deviation of total value iteration costs per episode over {num_episodes} episode(s): {value_iteration_costs_per_episode.std()}\n")

    print(f"Initial state of episode #{specified_episode}: {lqr_states[specified_episode, 0]}")
    print(f"Final LQR state of episode #{specified_episode}: {lqr_states[specified_episode, -1]}")
    print(f"Final value iteration state of episode #{specified_episode}: {value_iteration_states[specified_episode, -1]}\n")

    print(f"Reference state: {reference_point[:, 0]}\n")

    print(f"Difference between final LQR state of episode #{specified_episode} and reference state: {np.abs(lqr_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Norm between final LQR state of episode #{specified_episode} and reference state: {np.linalg.norm(lqr_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Difference between final value iteration state of episode #{specified_episode} and reference state: {np.abs(value_iteration_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Norm between final value iteration state of episode #{specified_episode} and reference state: {np.linalg.norm(value_iteration_states[specified_episode, -1] - reference_point[:, 0])}\n")

    # Plot dynamics over time for all state dimensions
    ax = fig.add_subplot(3, 3, 2)
    ax.set_title(f"Dynamics Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State value")
    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        ax.plot(lqr_states[specified_episode, :, i])
    ax.legend(labels)
    for i in range(state_dim):
        ax.scatter(step_range, lqr_states[specified_episode, :, i])

    # Plot dynamics over time for all state dimensions
    ax = fig.add_subplot(3, 3, 3)
    ax.set_title(f"Dynamics Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State value")
    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        ax.plot(value_iteration_states[specified_episode, :, i])
    ax.legend(labels)
    for i in range(state_dim):
        ax.scatter(step_range, value_iteration_states[specified_episode, :, i])

    # Plot x_0 vs x_1 vs x_2
    ax = fig.add_subplot(3, 3, 4, projection='3d')
    ax.set_title(f"Controllers in Environment (3D; Episode #{specified_episode})")
    ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
    ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
    ax.set_zlim(state_minimums[2, 0], state_maximums[2, 0])
    ax.plot3D(
        lqr_states[specified_episode, :, 0],
        lqr_states[specified_episode, :, 1],
        lqr_states[specified_episode, :, 2]
    )
    ax.plot3D(
        value_iteration_states[specified_episode, :, 0],
        value_iteration_states[specified_episode, :, 1],
        value_iteration_states[specified_episode, :, 2]
    )

    # Plot histogram of actions over time
    ax = fig.add_subplot(3, 3, 5)
    ax.set_title(f"Histogram of Actions Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Action Value")
    ax.set_ylabel("Frequency")
    ax.hist(lqr_actions[specified_episode, :, 0])
    ax.hist(value_iteration_actions[specified_episode, :, 0])

    # Plot scatter plot of actions over time
    ax = fig.add_subplot(3, 3, 6)
    ax.set_title(f"Scatter Plot of Actions Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Step #")
    ax.set_ylabel("Action Value")
    ax.scatter(step_range, lqr_actions[specified_episode, :, 0], s=5)
    ax.scatter(step_range, value_iteration_actions[specified_episode, :, 0], s=5)

    # Plot bellman errors over time
    ax = fig.add_subplot(3, 3, 8)
    ax.set_title(f"Bellman Errors During Policy Training")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Bellman Error")
    ax.plot(np.load(f"{koopman_policy.save_data_path}/training_data/bellman_errors.npy"))

    # Show/save plots
    save_figure("lorenz_dynamics_with_value_iteration_vs_lqr")
    # show_plot()

if __name__ == '__main__':
    print("\nTesting learned policy...\n")
    # watch_agent(num_episodes=100, num_steps_per_episode=int(20 / dt), specified_episode=42)
    watch_agent(num_episodes=1, num_steps_per_episode=int(20 / dt), specified_episode=0)