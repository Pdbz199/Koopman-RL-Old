# Imports
import matplotlib.pyplot as plt
import numpy as np

seed = 123
np.random.seed(seed)

from cost import reference_point, cost
from dynamics import state_minimums, state_maximums, state_dim, action_minimums, action_maximums, action_dim, state_order, action_order, all_actions, f

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
gamma = 0.99
reg_lambda = 1.0

plot_path = 'output/discrete_value_iteration/'
plot_file_extensions = ['svg', 'png']

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

# Koopman value iteration policy
koopman_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    'saved_models/linear-system-discrete-value-iteration-policy.pt',
    seed=seed
)

# Train Koopman policy
koopman_policy.train(training_epochs=2000, batch_size=2**12)

# Test policies
def watch_agent(num_episodes, step_limit, specifiedEpisode):
    np.random.seed(seed)

    if specifiedEpisode is None:
        specifiedEpisode = num_episodes-1

    states = np.zeros([num_episodes,step_limit,state_dim])
    actions = np.zeros([num_episodes,step_limit,action_dim])
    costs = np.zeros([num_episodes])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    ).T

    for episode in range(num_episodes):
        state = np.vstack(initial_states[episode])

        cumulative_cost = 0

        for step in range(step_limit):
            states[episode,step] = state[:,0]

            action = koopman_policy.get_action(state)
            actions[episode,step] = action

            cumulative_cost += cost(state, action)[0,0]

            state = f(state, action)

        costs[episode] = cumulative_cost

    print(f"Mean cost per episode over {num_episodes} episode(s): {np.mean(costs)}\n")

    print(f"Initial state of episode #{specifiedEpisode}: {states[specifiedEpisode,0]}")
    print(f"Final state of episode #{specifiedEpisode}: {states[specifiedEpisode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state: {np.abs(states[specifiedEpisode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state: {np.linalg.norm(states[specifiedEpisode,-1] - reference_point[:,0])}")

    # Plot dynamics over time for all state dimensions
    plt.title("Dynamics Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("State value")

    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        plt.plot(states[specifiedEpisode,:,i], label=labels[i])
    plt.legend(labels)
    plt.tight_layout()
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'states-over-time-2d.' + plot_file_extension)
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
        states[specifiedEpisode,:,2]
    )
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'states-over-time-3d.' + plot_file_extension)
    # plt.show()
    plt.clf()

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(actions[specifiedEpisode,:,0])
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'actions-histogram.' + plot_file_extension)
    # plt.show()
    plt.clf()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(actions.shape[1]), actions[specifiedEpisode,:,0], s=5)
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'actions-scatter-plot.' + plot_file_extension)
    # plt.show()
    plt.clf()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=200, specifiedEpisode=42)