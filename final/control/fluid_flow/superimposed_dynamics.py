# Imports
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from cost import cost, reference_point
from dynamics import action_dim, all_actions, action_order, dt, f, state_dim, state_minimums, state_maximums, state_order

sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy

# Dummy Koopman tensor
N = 10
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

# Koopman value iteration policy
gamma = 0.99
reg_lambda = 1.0
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
    'saved_models/fluid-flow-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.003,
    load_model=True
)

# Test policies
def watch_agent(num_episodes, step_limit, specified_episode=None):
    np.random.seed(seed)

    if specified_episode is None:
        specified_episode = num_episodes-1

    uncontrolled_states = np.zeros([num_episodes,step_limit,state_dim])
    uncontrolled_actions = np.zeros([num_episodes,step_limit,action_dim])
    uncontrolled_costs = np.zeros([num_episodes,step_limit])

    controlled_states = np.zeros([num_episodes,step_limit,state_dim])
    controlled_actions = np.zeros([num_episodes,step_limit,action_dim])
    controlled_costs = np.zeros([num_episodes,step_limit])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    ).T

    for episode in range(num_episodes):
        state = np.vstack(initial_states[episode])

        uncontrolled_state = state
        controlled_state = state

        for step in range(step_limit):
            uncontrolled_states[episode,step] = uncontrolled_state[:,0]
            controlled_states[episode,step] = controlled_state[:,0]

            uncontrolled_action = np.zeros([action_dim,1])
            uncontrolled_actions[episode,step] = uncontrolled_action
            controlled_action, _ = koopman_policy.get_action(controlled_state)
            controlled_actions[episode,step] = controlled_action

            uncontrolled_costs[episode,step] = cost(uncontrolled_state, uncontrolled_action)[0,0]
            controlled_costs[episode,step] = cost(controlled_state, controlled_action)[0,0]

            uncontrolled_state = f(uncontrolled_state, uncontrolled_action)
            controlled_state = f(controlled_state, controlled_action)

    plt.title("Total Cost Per Episode")
    plt.xlabel("Episode #")
    plt.ylabel("Total Cost")
    plt.plot(uncontrolled_costs.sum(1))
    plt.plot(controlled_costs.sum(1))
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'total-cost-per-episode.' + plot_file_extension)
    plt.show()
    # plt.clf()

    print(f"Mean of total costs per episode over {num_episodes} episode(s): {controlled_costs.sum(1).mean()}")
    print(f"Standard deviation of total costs per episode over {num_episodes} episode(s): {controlled_costs.sum(1).std()}\n")

    print(f"Initial state of episode #{specified_episode}: {controlled_states[specified_episode,0]}")
    print(f"Final state of episode #{specified_episode}: {controlled_states[specified_episode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specified_episode} and reference state: {np.abs(controlled_states[specified_episode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specified_episode} and reference state: {np.linalg.norm(controlled_states[specified_episode,-1] - reference_point[:,0])}")

    # Plot dynamics over time for all state dimensions
    plt.title("Dynamics Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("State value")

    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        plt.plot(controlled_states[specified_episode,:,i], label=labels[i])
    plt.legend(labels)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'states-over-time-2d.' + plot_file_extension)
    plt.show()
    # plt.clf()

    # Plot x_0 vs x_1 vs x_2
    ax = plt.axes(projection='3d')
    ax.set_title(f"Controllers in Environment (3D; Episode #{specified_episode})")
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
    ax.plot3D(
        uncontrolled_states[specified_episode,:,0],
        uncontrolled_states[specified_episode,:,1],
        uncontrolled_states[specified_episode,:,2],
        'gray'
    )
    ax.plot3D(
        controlled_states[specified_episode,:,0],
        controlled_states[specified_episode,:,1],
        controlled_states[specified_episode,:,2],
        'blue'
    )
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'states-over-time-3d.' + plot_file_extension)
    plt.show()
    # plt.clf()

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specified_episode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(uncontrolled_actions[specified_episode,:,0])
    plt.hist(controlled_actions[specified_episode,:,0])
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'actions-histogram.' + plot_file_extension)
    plt.show()
    # plt.clf()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specified_episode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(uncontrolled_actions.shape[1]), uncontrolled_actions[specified_episode,:,0], s=5)
    plt.scatter(np.arange(controlled_actions.shape[1]), controlled_actions[specified_episode,:,0], s=5)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'actions-scatter-plot.' + plot_file_extension)
    plt.show()
    # plt.clf()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=1, step_limit=int(25.0 / dt), specified_episode=0)