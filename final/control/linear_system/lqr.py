# Imports
import matplotlib.pyplot as plt
import numpy as np

from cost import Q, R, reference_point, cost
from dynamics import state_dim, action_dim, state_minimums, state_maximums, A, B, f

import sys
sys.path.append('../../../')
from final.control.policies.lqr import LQRPolicy

# Set seed
seed = 123
np.random.seed(seed)

# Variables
gamma = 0.99
reg_lambda = 1.0

plot_path = 'output/lqr/'
plot_file_extensions = ['.svg', '.png']

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

# Test policy
def watch_agent(num_episodes, step_limit, specifiedEpisode=None):
    if specifiedEpisode is None:
        specifiedEpisode = num_episodes-1

    states = np.zeros([num_episodes,step_limit,state_dim])
    actions = np.zeros([num_episodes,step_limit,action_dim])
    costs = np.zeros([num_episodes])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, num_episodes]
    ).T

    for episode in range(num_episodes):
        state = np.vstack(initial_states[episode])

        cumulative_cost = 0

        for step in range(step_limit):
            states[episode,step] = state[:,0]

            action = lqr_policy.get_action(state)
            # if action[0,0] > action_range:
            #     action = np.array([[action_range]])
            # elif action[0,0] < -action_range:
            #     action = np.array([[-action_range]])
            actions[episode,step] = action

            cumulative_cost += cost(state, action)[0,0]

            state = f(state, action)

        costs[episode] = cumulative_cost

    print(f"Mean cost per episode over {num_episodes} episode(s): {np.mean(costs)}\n")

    print(f"Initial state of episode #{specifiedEpisode}: {states[specifiedEpisode,0]}")
    print(f"Final state of episode #{specifiedEpisode}: {states[specifiedEpisode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state: {np.abs(states[specifiedEpisode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state: {np.linalg.norm(states[specifiedEpisode,-1] - reference_point[:,0])}\n")

    # Plot dynamics over time for all state dimensions for both controllers
    plt.title("Dynamics Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("State value")

    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        plt.plot(states[specifiedEpisode,:,i], label=labels[i])
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(plot_path + 'states-over-time-2d' + plot_file_extensions[0])
    plt.savefig(plot_path + 'states-over-time-2d' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

    # Plot x_0 vs x_1 vs x_2
    ax = plt.axes(projection='3d')
    ax.set_title(f"Controllers in Environment (3D; Episode #{specifiedEpisode})")
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
    ax.plot3D(
        states[specifiedEpisode,:,0],
        states[specifiedEpisode,:,1],
        states[specifiedEpisode,:,2],
        'gray'
    )
    plt.savefig(plot_path + 'states-over-time-3d' + plot_file_extensions[0])
    plt.savefig(plot_path + 'states-over-time-3d' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(actions[specifiedEpisode,:,0])
    plt.savefig(plot_path + 'actions-histogram' + plot_file_extensions[0])
    plt.savefig(plot_path + 'actions-histogram' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(actions.shape[1]), actions[specifiedEpisode,:,0], s=5)
    plt.savefig(plot_path + 'actions-scatter-plot' + plot_file_extensions[0])
    plt.savefig(plot_path + 'actions-scatter-plot' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=200, specifiedEpisode=42)