# Imports
import matplotlib.pyplot as plt
import numpy as np

seed = 123
np.random.seed(seed)

from cost import reference_point, cost
from dynamics import (
    dt,
    state_dim,
    action_dim,
    state_minimums,
    state_maximums,
    x_e,
    y_e,
    z_e,
    all_actions,
    state_order,
    action_order,
    random_policy,
    f
)

import sys
sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.continuous_actor_critic import ContinuousKoopmanPolicyIterationPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0

plot_path = 'output/continuous_actor_critic/'
plot_file_extensions = ['svg', 'png']

#%% Generate path-based data
num_episodes = 500
num_steps_per_episode = int(25.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints

X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
)

for episode in range(num_episodes):
    x = np.vstack(initial_states[:,episode])
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
deviation = 2.0
state_minimums = np.array([
    [x_e-deviation],
    [y_e-deviation],
    [z_e-deviation]
])
state_maximums = np.array([
    [x_e+deviation],
    [y_e+deviation],
    [z_e+deviation]
])
local_state_minimums = np.array([
    [x_e-deviation],
    [y_e-deviation],
    [z_e-deviation]
])
local_state_maximums = np.array([
    [x_e+deviation],
    [y_e+deviation],
    [z_e+deviation]
])

# all_actions = np.arange(-75, 75+1.0, 1.0)
# all_actions = np.round(all_actions, decimals=2)
# all_actions = np.array([all_actions])

koopman_policy = ContinuousKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    # state_minimums,
    # state_maximums,
    local_state_minimums,
    local_state_maximums,
    all_actions,
    cost,
    'saved_models/lorenz-continuous-actor-critic-policy.pt',
    # use_ols=False,
    learning_rate=0.003,
    dt=dt,
    seed=seed,
    load_model=True
)
print(f"\nLearning rate: {koopman_policy.learning_rate}\n")

# Train Koopman policy
# koopman_policy.train(num_training_episodes=2000, num_steps_per_episode=int(25.0 / dt))

# Test policies
def watch_agent(num_episodes, step_limit, specifiedEpisode=None):
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

            action, _ = koopman_policy.get_action(state)
            action = np.array([action.numpy()])
            actions[episode,step] = action

            cumulative_cost += cost(state, action)[0,0]

            state = f(state, action)

        costs[episode] = cumulative_cost

    print(f"Mean cost per episode over {num_episodes} episode(s) (Koopman controller): {np.mean(costs)}\n")

    print(f"Initial state of episode #{specifiedEpisode} (Koopman controller): {states[specifiedEpisode,0]}")
    print(f"Final state of episode #{specifiedEpisode} (Koopman controller): {states[specifiedEpisode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {np.abs(states[specifiedEpisode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {np.linalg.norm(states[specifiedEpisode,-1] - reference_point[:,0])}")

    # Plot dynamics over time for all state dimensions for both controllers
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

    # Plot x_0 vs x_1 vs x_2 for both controller types
    ax = plt.axes(projection='3d')
    ax.set_title(f"Controllers in Environment (3D; Episode #{specifiedEpisode})")
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
    ax.plot(
        states[specifiedEpisode,:,0],
        states[specifiedEpisode,:,1],
        states[specifiedEpisode,:,2],
        'gray'
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
watch_agent(num_episodes=100, step_limit=int(25.0 / dt), specifiedEpisode=42)