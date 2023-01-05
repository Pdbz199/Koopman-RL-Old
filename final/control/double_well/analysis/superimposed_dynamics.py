# Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from cost import (
    cost,
    reference_point,
    R,
    Q
)
from dynamics import (
    action_dim,
    all_actions,
    continuous_A,
    continuous_B,
    dt,
    f,
    random_policy,
    state_dim,
    state_minimums,
    state_maximums,
    zero_policy
)

sys.path.append('../../../')
# import final.observables as observables
# from final.tensor import KoopmanTensor
from final.control.policies.lqr import LQRPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy

# Dummy Koopman tensor
# N = 10
# X = np.zeros([state_dim,N])
# Y = np.zeros([state_dim,N])
# U = np.zeros([action_dim,N])
# tensor = KoopmanTensor(
#     X,
#     Y,
#     U,
#     phi=observables.monomials(state_order),
#     psi=observables.monomials(action_order),
#     regressor='ols'
# )

folder = "./analysis/tmp"
with open(f'{folder}/shotgun-tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

# Controller params
gamma = 0.99
reg_lambda = 1.0

# LQR policy
lqr_policy = LQRPolicy(
    continuous_A,
    continuous_B,
    Q,
    R,
    reference_point,
    gamma,
    reg_lambda,
    dt=dt,
    seed=seed
)

# Koopman value iteration policy
value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    'saved_models/double-well-discrete-value-iteration-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.0003,
    load_model=True
)

# Koopman actor critic policy
actor_critic_policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'saved_models/double-well-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.0003,
    load_model=True
)

# Test policies
def watch_agent(num_episodes, step_limit, specified_episode=None):
    np.random.seed(seed)

    if specified_episode is None:
        specified_episode = num_episodes-1

    lqr_states = np.zeros([num_episodes,step_limit,state_dim])
    lqr_actions = np.zeros([num_episodes,step_limit,action_dim])
    lqr_costs = np.zeros([num_episodes,step_limit])

    actor_critic_states = np.zeros([num_episodes,step_limit,state_dim])
    actor_critic_actions = np.zeros([num_episodes,step_limit,action_dim])
    actor_critic_costs = np.zeros([num_episodes,step_limit])

    value_iteration_states = np.zeros([num_episodes,step_limit,state_dim])
    value_iteration_actions = np.zeros([num_episodes,step_limit,action_dim])
    value_iteration_costs = np.zeros([num_episodes,step_limit])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    ).T

    for episode in range(num_episodes):
        state = np.vstack(initial_states[episode])

        lqr_state = state
        actor_critic_state = state
        value_iteration_state = state

        for step in range(step_limit):
            lqr_states[episode,step] = lqr_state[:,0]
            actor_critic_states[episode,step] = actor_critic_state[:,0]
            value_iteration_states[episode,step] = value_iteration_state[:,0]

            # Zero policies
            # lqr_action = zero_policy(lqr_state)
            # actor_critic_action = zero_policy(actor_critic_state)
            # value_iteration_action = zero_policy(value_iteration_state)

            # Random policies
            # lqr_action = random_policy(lqr_state)
            # actor_critic_action = random_policy(actor_critic_state)
            # value_iteration_action = random_policy(value_iteration_state)

            # Learned policies
            lqr_action = lqr_policy.get_action(lqr_state)
            actor_critic_action, _ = actor_critic_policy.get_action(actor_critic_state)
            value_iteration_action = value_iteration_policy.get_action(value_iteration_state)

            # Save actions
            lqr_actions[episode,step] = lqr_action
            actor_critic_actions[episode,step] = actor_critic_action
            value_iteration_actions[episode,step] = value_iteration_action

            # Save costs
            lqr_costs[episode,step] = cost(lqr_state, lqr_action)[0,0]
            actor_critic_costs[episode,step] = cost(actor_critic_state, actor_critic_action)[0,0]
            value_iteration_costs[episode,step] = cost(value_iteration_state, value_iteration_action)[0,0]

            # Update state
            lqr_state = f(lqr_state, lqr_action)
            actor_critic_state = f(actor_critic_state, actor_critic_action)
            value_iteration_state = f(value_iteration_state, value_iteration_action)

    controller_labels = ['LQR', 'Actor Critic', 'Value Iteration']

    plt.title("Total Cost Per Episode")
    plt.xlabel("Episode #")
    plt.ylabel("Total Cost")
    plt.plot(lqr_costs[specified_episode])
    plt.plot(actor_critic_costs[specified_episode])
    plt.plot(value_iteration_costs[specified_episode])
    plt.legend(controller_labels)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'total-cost-per-episode.' + plot_file_extension)
    plt.show()
    # plt.clf()

    print(f"Mean of total costs per episode over {num_episodes} episode(s) (LQR): {lqr_costs.sum(1).mean()}")
    print(f"Mean of total costs per episode over {num_episodes} episode(s) (Actor Critic): {actor_critic_costs.sum(1).mean()}")
    print(f"Mean of total costs per episode over {num_episodes} episode(s) (Value Iteration): {value_iteration_costs.sum(1).mean()}\n")

    print(f"Standard deviation of total costs per episode over {num_episodes} episode(s) (LQR): {lqr_costs.sum(1).std()}")
    print(f"Standard deviation of total costs per episode over {num_episodes} episode(s) (Actor Critic): {actor_critic_costs.sum(1).std()}")
    print(f"Standard deviation of total costs per episode over {num_episodes} episode(s) (Value Iteration): {value_iteration_costs.sum(1).std()}\n")

    print(f"Initial state of episode #{specified_episode} (LQR): {lqr_states[specified_episode,0]}")
    print(f"Initial state of episode #{specified_episode} (Actor Critic): {actor_critic_states[specified_episode,0]}")
    print(f"Initial state of episode #{specified_episode} (Value Iteration): {value_iteration_states[specified_episode,0]}\n")

    print(f"Final state of episode #{specified_episode} (LQR): {lqr_states[specified_episode,-1]}")
    print(f"Final state of episode #{specified_episode} (Actor Critic): {actor_critic_states[specified_episode,-1]}")
    print(f"Final state of episode #{specified_episode} (Value Iteration): {value_iteration_states[specified_episode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specified_episode} and reference state (LQR): {np.abs(lqr_states[specified_episode,-1] - reference_point[:,0])}")
    print(f"Difference between final state of episode #{specified_episode} and reference state (Actor Critic): {np.abs(actor_critic_states[specified_episode,-1] - reference_point[:,0])}")
    print(f"Difference between final state of episode #{specified_episode} and reference state (Value Iteration): {np.abs(value_iteration_states[specified_episode,-1] - reference_point[:,0])}\n")

    print(f"Norm between final state of episode #{specified_episode} and reference state (LQR): {np.linalg.norm(lqr_states[specified_episode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specified_episode} and reference state (Actor Critic): {np.linalg.norm(actor_critic_states[specified_episode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specified_episode} and reference state (Value Iteration): {np.linalg.norm(value_iteration_states[specified_episode,-1] - reference_point[:,0])}\n")

    # Plot dynamics over time for all state dimensions
    plt.title("Dynamics Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("State value")

    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        plt.plot(lqr_states[specified_episode,:,i], label=labels[i])
    plt.legend(labels)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'states-over-time.' + plot_file_extension)
    plt.show()
    # plt.clf()

    # Plot x_0 vs x_1
    plt.suptitle(f"Controllers in Environment (2D; Episode #{specified_episode})")
    plt.xlim(state_minimums[0,0], state_maximums[0,0])
    plt.ylim(state_minimums[1,0], state_maximums[1,0])
    plt.plot(
        lqr_states[specified_episode,:,0],
        lqr_states[specified_episode,:,1]
    )
    plt.plot(
        actor_critic_states[specified_episode,:,0],
        actor_critic_states[specified_episode,:,1]
    )
    plt.plot(
        value_iteration_states[specified_episode,:,0],
        value_iteration_states[specified_episode,:,1]
    )
    plt.legend(controller_labels)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'x0-vs-x1.' + plot_file_extension)
    plt.show()
    # plt.clf()

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specified_episode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(lqr_actions[specified_episode,:,0])
    plt.hist(actor_critic_actions[specified_episode,:,0])
    plt.hist(value_iteration_actions[specified_episode,:,0])
    plt.legend(controller_labels)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
    #     plt.savefig(plot_path + 'actions-histogram.' + plot_file_extension)
    plt.show()
    # plt.clf()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specified_episode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(lqr_actions.shape[1]), lqr_actions[specified_episode,:,0], s=5)
    plt.scatter(np.arange(actor_critic_actions.shape[1]), actor_critic_actions[specified_episode,:,0], s=5)
    plt.scatter(np.arange(value_iteration_actions.shape[1]), value_iteration_actions[specified_episode,:,0], s=5)
    plt.legend(controller_labels)
    plt.tight_layout()
    # for plot_file_extension in plot_file_extensions:
        # plt.savefig(plot_path + 'actions-scatter-plot.' + plot_file_extension)
    plt.show()
    # plt.clf()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=5, step_limit=int(25.0 / dt), specified_episode=-1)