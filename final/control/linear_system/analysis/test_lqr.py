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
    f,
    state_dim,
    state_maximums,
    state_minimums,
    zero_policy
)

sys.path.append('../../../')

#%% Load LQR policy
with open('./analysis/tmp/lqr/policy.pickle', 'rb') as handle:
    lqr_policy = pickle.load(handle)

#%% Functions for showing/saving figures
def show_plot():
    plt.tight_layout()
    plt.show()

path = "./analysis/tmp/lqr/plots"
plot_file_extensions = ['png', 'svg']
def save_figure(title: str):
    plt.tight_layout()
    for plot_file_extension in plot_file_extensions:
        plt.savefig(f"{path}/{title}.{plot_file_extension}")
    plt.clf()

#%% Test policy
def watch_agent(num_episodes, num_steps_per_episode, specified_episode=None):
    if specified_episode is None:
        specified_episode = num_episodes-1

    uncontrolled_states = np.zeros((num_episodes, num_steps_per_episode, state_dim))
    uncontrolled_actions = np.zeros((num_episodes, num_steps_per_episode, action_dim))
    uncontrolled_costs = np.zeros((num_episodes, num_steps_per_episode))

    controlled_states = np.zeros_like(uncontrolled_states)
    controlled_actions = np.zeros_like(uncontrolled_actions)
    controlled_costs = np.zeros_like(uncontrolled_costs)

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, num_episodes]
    ).T

    for episode_num in range(num_episodes):
        # Extract initial state
        state = np.vstack(initial_states[episode_num])

        # Set un/controlled states to initial state
        uncontrolled_state = state
        controlled_state = state

        for step_num in range(num_steps_per_episode):
            # Save states to arrays
            uncontrolled_states[episode_num, step_num] = uncontrolled_state[:, 0]
            controlled_states[episode_num, step_num] = controlled_state[:, 0]

            # Get actions and save them to arrays
            uncontrolled_action = zero_policy(uncontrolled_state)
            uncontrolled_actions[episode_num, step_num] = uncontrolled_action[:, 0]
            controlled_action = lqr_policy.get_action(controlled_state)#, is_entropy_regularized=False)
            controlled_actions[episode_num, step_num] = controlled_action[:, 0]

            # Compute costs
            uncontrolled_costs[episode_num, step_num] = cost(uncontrolled_state, uncontrolled_action)[0, 0]
            controlled_costs[episode_num, step_num] = cost(controlled_state, controlled_action)[0, 0]

            # Compute next state
            uncontrolled_state = f(uncontrolled_state, uncontrolled_action)
            controlled_state = f(controlled_state, controlled_action)

    # Plot configuration
    fig = plt.figure(figsize=(16, 9))
    episode_range = np.arange(num_episodes)
    step_range = np.arange(num_steps_per_episode)
    types_of_control_labels = ['uncontrolled', 'controlled']

    # Plot total costs per episode
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title("Total Cost Per Episode")
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Total Cost")
    uncontrolled_costs_per_episode = uncontrolled_costs.sum(axis=1)
    controlled_costs_per_episode = controlled_costs.sum(axis=1)
    ax.plot(uncontrolled_costs_per_episode)
    ax.plot(controlled_costs_per_episode)
    ax.scatter(episode_range, uncontrolled_costs_per_episode)
    ax.scatter(episode_range, controlled_costs_per_episode)
    ax.legend(types_of_control_labels)

    # Print out specific values
    print(f"Mean of total uncontrolled costs per episode over {num_episodes} episode(s): {uncontrolled_costs_per_episode.mean()}")
    print(f"Standard deviation of total uncontrolled costs per episode over {num_episodes} episode(s): {uncontrolled_costs_per_episode.std()}")
    print(f"Mean of total controlled costs per episode over {num_episodes} episode(s): {controlled_costs_per_episode.mean()}")
    print(f"Standard deviation of total controlled costs per episode over {num_episodes} episode(s): {controlled_costs_per_episode.std()}\n")

    print(f"Initial state of episode #{specified_episode}: {uncontrolled_states[specified_episode, 0]}")
    print(f"Final uncontrolled state of episode #{specified_episode}: {uncontrolled_states[specified_episode, -1]}")
    print(f"Final controlled state of episode #{specified_episode}: {controlled_states[specified_episode, -1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final uncontrolled state of episode #{specified_episode} and reference state: {np.abs(uncontrolled_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Norm between final uncontrolled state of episode #{specified_episode} and reference state: {np.linalg.norm(uncontrolled_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Difference between final controlled state of episode #{specified_episode} and reference state: {np.abs(controlled_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Norm between final controlled state of episode #{specified_episode} and reference state: {np.linalg.norm(controlled_states[specified_episode, -1] - reference_point[:, 0])}\n")

    # Plot dynamics over time for all state dimensions for both controllers
    ax = fig.add_subplot(2, 3, 2)
    ax.set_title(f"Uncontrolled Dynamics Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("State value")
    # Create and assign labels as a function of number of dimensions of uncontrolled state
    state_labels = []
    for i in range(state_dim):
        state_labels.append(f"uncontrolled x_{i}")
        ax.plot(uncontrolled_states[specified_episode, :, i], label=state_labels[i])
    ax.legend(state_labels)
    for i in range(state_dim):
        ax.scatter(step_range, uncontrolled_states[specified_episode, :, i])

    ax = fig.add_subplot(2, 3, 3)
    ax.set_title(f"Controlled Dynamics Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("State value")
    # Create and assign labels as a function of number of dimensions of controlled state
    state_labels = []
    for i in range(state_dim):
        state_labels.append(f"controlled x_{i}")
        ax.plot(controlled_states[specified_episode, :, i])
    ax.legend(state_labels)
    for i in range(state_dim):
        ax.scatter(step_range, controlled_states[specified_episode, :, i])

    # Plot x_0 vs x_1 (2D)
    # ax = fig.add_subplot(2, 3, 4)
    # ax.set_title(f"Controllers in Environment (3D; Episode #{specified_episode})")
    # ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
    # ax.set_xlabel("x_0")
    # ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
    # ax.set_xlabel("x_1")
    # ax.plot(
    #     uncontrolled_states[specified_episode, :, 0],
    #     uncontrolled_states[specified_episode, :, 1]
    # )
    # ax.plot(
    #     controlled_states[specified_episode, :, 0],
    #     controlled_states[specified_episode, :, 1]
    # )

    # Plot x_0 vs x_1 vs x_2 (3D)
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.set_title(f"Controllers in Environment (3D; Episode #{specified_episode})")
    ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
    ax.set_xlabel("x_0")
    ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
    ax.set_ylabel("x_1")
    ax.set_zlim(state_minimums[2, 0], state_maximums[2, 0])
    ax.set_zlabel("x_2")
    ax.plot3D(
        uncontrolled_states[specified_episode, :, 0],
        uncontrolled_states[specified_episode, :, 1],
        uncontrolled_states[specified_episode, :, 2]
    )
    ax.plot3D(
        controlled_states[specified_episode, :, 0],
        controlled_states[specified_episode, :, 1],
        controlled_states[specified_episode, :, 2]
    )

    # Plot histogram of actions over time
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title(f"Histogram of Actions Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Action Value")
    ax.set_ylabel("Frequency")
    ax.hist(uncontrolled_actions[specified_episode, :, 0])
    ax.hist(controlled_actions[specified_episode, :, 0])
    ax.legend(types_of_control_labels)

    # Plot scatter plot of actions over time
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title(f"Scatter Plot of Actions Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Step #")
    ax.set_ylabel("Action Value")
    ax.scatter(step_range, uncontrolled_actions[specified_episode, :, 0], s=5)
    ax.scatter(step_range, controlled_actions[specified_episode, :, 0], s=5)
    ax.legend(types_of_control_labels)

    # Show plot
    save_figure("linear_system_dynamics_with_lqr")
    # show_plot()

if __name__ == '__main__':
    print("\nTesting learned policy...\n")
    watch_agent(num_episodes=100, num_steps_per_episode=200, specified_episode=42)