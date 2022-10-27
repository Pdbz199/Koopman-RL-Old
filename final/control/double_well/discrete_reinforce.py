# Imports
import matplotlib.pyplot as plt
import numpy as np

seed = 123
np.random.seed(seed)

from cost import cost, reference_point
from dynamics import action_dim, all_actions, action_order, dt, f, state_dim, state_minimums, state_maximums, state_order, random_policy

import sys
sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.discrete_reinforce import DiscreteKoopmanPolicyIterationPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0

plot_path = 'output/discrete_reinforce/'
plot_file_extensions = ['svg', 'png']

# Generate datasets
num_episodes = 500
num_steps_per_episode = int(10.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints

X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
).T

for episode in range(num_episodes):
    x = np.vstack(initial_states[episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = random_policy()
        U[:,(episode*num_steps_per_episode)+step] = u[:,0]
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

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
koopman_policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'saved_models/double-well-discrete-reinforce-policy.pt',
    dt=dt,
    seed=seed
)

# Train Koopman policy
koopman_policy.train(num_training_episodes=2000, num_steps_per_episode=int(25.0 / dt))

# Test policies
def watch_agent(num_episodes, step_limit, specifiedEpisode=None):
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

            action, _ = koopman_policy.get_action(state)
            action = np.array([[action]])
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
    plt.savefig(plot_path + 'states-over-time.' + plot_file_extensions[0])
    plt.savefig(plot_path + 'states-over-time.' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

    # Plot x_0 vs x_1
    plt.suptitle(f"Controllers in Environment (2D; Episode #{specifiedEpisode})")
    plt.xlim(state_minimums[0,0], state_maximums[0,0])
    plt.ylim(state_minimums[1,0], state_maximums[1,0])
    plt.plot(
        states[specifiedEpisode,:,0],
        states[specifiedEpisode,:,1],
        'gray'
    )
    plt.savefig(plot_path + 'x0-vs-x1.' + plot_file_extensions[0])
    plt.savefig(plot_path + 'x0-vs-x1.' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(actions[specifiedEpisode,:,0])
    plt.savefig(plot_path + 'actions-histogram.' + plot_file_extensions[0])
    plt.savefig(plot_path + 'actions-histogram.' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(actions.shape[1]), actions[specifiedEpisode,:,0], s=5)
    plt.savefig(plot_path + 'actions-scatter-plot.' + plot_file_extensions[0])
    plt.savefig(plot_path + 'actions-scatter-plot.' + plot_file_extensions[1])
    # plt.show()
    plt.clf()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=int(25.0 / dt), specifiedEpisode=42)