#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import torch

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from cost import cost, reference_point
from dynamics import (
    action_maximums,
    action_minimums,
    action_dim,
    all_actions,
    dt,
    f,
    state_dim,
    state_maximums,
    state_minimums
)

sys.path.append('../../../')

#%% Load Koopman tensor with pickle
# with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
#     tensor = pickle.load(handle)

#%% Load LQR policy
with open('./analysis/tmp/lqr/policy.pickle', 'rb') as file:
    lqr_policy = pickle.load(file)

#%% Variables
gamma = 0.99
reg_lambda = 1.0

#%% Koopman actor-critic policy
with open('./analysis/tmp/continuous_actor_critic/policy.pkl', 'rb') as file:
    koopman_policy = pickle.load(file)

#%% Functions for showing/saving figures
def show_plot():
    plt.tight_layout()
    plt.show()

path = "./analysis/tmp/continuous_actor_critic/plots"
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

    actor_critic_states = np.zeros_like(lqr_states)
    actor_critic_actions = np.zeros_like(lqr_actions)
    actor_critic_costs = np.zeros_like(lqr_costs)

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
        actor_critic_state = state

        for step_num in range(num_steps_per_episode):
            # Save latest states
            lqr_states[episode_num,step_num] = lqr_state[:, 0]
            actor_critic_states[episode_num,step_num] = actor_critic_state[:, 0]

            # Get actions for current state and save them
            lqr_action = lqr_policy.get_action(lqr_state)
            lqr_actions[episode_num, step_num] = lqr_action[:, 0]
            actor_critic_action = koopman_policy.agent.act(torch.Tensor(actor_critic_state[:, 0])).numpy()
            actor_critic_action = np.vstack(actor_critic_action)
            actor_critic_actions[episode_num, step_num] = actor_critic_action[:, 0]

            # Compute costs
            lqr_costs[episode_num,step_num] = cost(lqr_state, lqr_action)[0, 0]
            actor_critic_costs[episode_num,step_num] = cost(actor_critic_state, actor_critic_action)[0, 0]

            # Compute next states
            lqr_state = f(lqr_state, lqr_action)
            actor_critic_state = f(actor_critic_state, actor_critic_action)

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
    actor_critic_costs_per_episode = actor_critic_costs.sum(axis=1)
    ax.plot(lqr_costs_per_episode)
    ax.plot(actor_critic_costs_per_episode)
    ax.scatter(episode_range, lqr_costs_per_episode)
    ax.scatter(episode_range, actor_critic_costs_per_episode)

    print(f"Mean of total LQR costs per episode over {num_episodes} episode(s): {lqr_costs_per_episode.mean()}")
    print(f"Standard deviation of total LQR costs per episode over {num_episodes} episode(s): {lqr_costs_per_episode.std()}")
    print(f"Mean of total continuous actor critic costs per episode over {num_episodes} episode(s): {actor_critic_costs_per_episode.mean()}")
    print(f"Standard deviation of total continuous actor critic costs per episode over {num_episodes} episode(s): {actor_critic_costs_per_episode.std()}\n")

    print(f"Initial state of episode #{specified_episode}: {lqr_states[specified_episode, 0]}")
    print(f"Final LQR state of episode #{specified_episode}: {lqr_states[specified_episode, -1]}")
    print(f"Final continuous actor critic state of episode #{specified_episode}: {actor_critic_states[specified_episode, -1]}\n")

    print(f"Reference state: {reference_point[:, 0]}\n")

    print(f"Difference between final LQR state of episode #{specified_episode} and reference state: {np.abs(lqr_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Norm between final LQR state of episode #{specified_episode} and reference state: {np.linalg.norm(lqr_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Difference between final continuous actor critic state of episode #{specified_episode} and reference state: {np.abs(actor_critic_states[specified_episode, -1] - reference_point[:, 0])}")
    print(f"Norm between final continuous actor critic state of episode #{specified_episode} and reference state: {np.linalg.norm(actor_critic_states[specified_episode, -1] - reference_point[:, 0])}\n")

    # Plot dynamics over time for all state dimensions
    ax = fig.add_subplot(3, 3, 2)
    ax.set_title("Dynamics Over Time")
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
    ax.set_title("Dynamics Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State value")
    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        ax.plot(actor_critic_states[specified_episode, :, i])
    ax.legend(labels)
    for i in range(state_dim):
        ax.scatter(step_range, actor_critic_states[specified_episode, :, i])

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
        actor_critic_states[specified_episode, :, 0],
        actor_critic_states[specified_episode, :, 1],
        actor_critic_states[specified_episode, :, 2]
    )

    # Plot histogram of actions over time
    ax = fig.add_subplot(3, 3, 5)
    ax.set_title(f"Histogram of Actions Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Action Value")
    ax.set_ylabel("Frequency")
    ax.hist(lqr_actions[specified_episode, :, 0])
    ax.hist(actor_critic_actions[specified_episode, :, 0])

    # Plot scatter plot of actions over time
    ax = fig.add_subplot(3, 3, 6)
    ax.set_title(f"Scatter Plot of Actions Over Time (Episode #{specified_episode})")
    ax.set_xlabel("Step #")
    ax.set_ylabel("Action Value")
    ax.scatter(step_range, lqr_actions[specified_episode, :, 0], s=5)
    ax.scatter(step_range, actor_critic_actions[specified_episode, :, 0], s=5)

    # Plot total cost per episode during training
    # ax = fig.add_subplot(3, 3, 8)
    # ax.set_title("Total Cost Per Episode During Training")
    # ax.set_xlabel(f"Episode #{specified_episode}")
    # ax.set_ylabel("Cost")
    # ax.plot(-np.load(f"./analysis/tmp/continuous_actor_critic/training_data/rewards.npy").sum(axis=1))

    # Show/save plots
    save_figure("linear_system_dynamics_with_actor_critic_vs_lqr")
    # show_plot()

if __name__ == '__main__':
    print("\nTesting learned policy...\n")
    watch_agent(num_episodes=100, num_steps_per_episode=int(200 / dt), specified_episode=42)