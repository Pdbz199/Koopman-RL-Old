# Imports
import matplotlib.pyplot as plt
import numpy as np

from cost import Q, R, reference_point, cost
from dynamics import state_dim, action_dim, state_column_shape, dt, continuous_f, random_policy, f, continuous_A, continuous_B, state_order, action_order, all_actions, state_minimums, state_maximums
from scipy.integrate import solve_ivp

import sys
sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.lqr import LQRPolicy
from final.control.policies.continuous_reinforce import ContinuousKoopmanPolicyIterationPolicy

# Set seed
seed = 123
np.random.seed(seed)

# Variables
gamma = 0.99
reg_lambda = 1.0

plot_path = 'plots/continuous_reinforce/'
plot_file_extension = '.svg'

# LQR Policy
lqr_policy = LQRPolicy(
    continuous_A,
    continuous_B,
    Q,
    R,
    reference_point,
    gamma,
    reg_lambda,
    dt=dt,
    is_continuous=True,
    seed=seed
)

# Generate data
num_episodes = 500
num_steps_per_episode = int(50.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_states = np.zeros([num_episodes, state_dim])
for episode in range(num_episodes):
    x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 50.0], y0=x[:,0], method='RK45')
    initial_states[episode] = soln.y[:,-1]

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
koopman_policy = ContinuousKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'saved_models/fluid-flow-continuous-reinforce-policy.pt',
    learning_rate=0.0003,
    dt=dt,
    seed=seed
)

# Train Koopman policy
koopman_policy.train(num_training_episodes=1000, num_steps_per_episode=int(25.0 / dt))

# Test policies
def watch_agent(num_episodes, step_limit, specifiedEpisode=None):
    if specifiedEpisode is None:
        specifiedEpisode = num_episodes-1

    lqr_states = np.zeros([num_episodes,step_limit,state_dim])
    lqr_actions = np.zeros([num_episodes,step_limit,action_dim])
    lqr_costs = np.zeros([num_episodes])

    koopman_states = np.zeros([num_episodes,step_limit,state_dim])
    koopman_actions = np.zeros([num_episodes,step_limit,action_dim])
    koopman_costs = np.zeros([num_episodes])

    initial_states = np.zeros([num_episodes, state_dim])
    for episode in range(num_episodes):
        x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
        u = np.array([[0]])
        soln = solve_ivp(fun=continuous_f(u), t_span=[0, 50.0], y0=x[:,0], method='RK45')
        initial_states[episode] = soln.y[:,-1]

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

            koopman_action, _ = koopman_policy.get_action(koopman_state)
            koopman_action = np.array([[koopman_action]])
            koopman_actions[episode,step] = koopman_action

            lqr_cumulative_cost += cost(lqr_state, lqr_action)[0,0]
            koopman_cumulative_cost += cost(koopman_state, koopman_action)[0,0]

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
    plt.savefig(plot_path + 'states-over-time-2d' + plot_file_extension)
    plt.show()

    # Plot x_0 vs x_1 vs x_2 for both controller types
    fig = plt.figure()
    fig.suptitle(f"Controllers in Environment (3D; Episode #{specifiedEpisode})")

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("LQR Controller")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.plot(
        lqr_states[specifiedEpisode,:,0],
        lqr_states[specifiedEpisode,:,1],
        lqr_states[specifiedEpisode,:,2],
        'gray'
    )

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Koopman Controller")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.plot(
        koopman_states[specifiedEpisode,:,0],
        koopman_states[specifiedEpisode,:,1],
        koopman_states[specifiedEpisode,:,2],
        'gray'
    )

    plt.savefig(plot_path + 'states-over-time-3d' + plot_file_extension)
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
    plt.savefig(plot_path + 'actions-histogram' + plot_file_extension)
    plt.show()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(lqr_actions.shape[1]), lqr_actions[specifiedEpisode,:,0], s=5)
    plt.scatter(np.arange(koopman_actions.shape[1]), koopman_actions[specifiedEpisode,:,0], s=5)
    plt.legend(labels)
    plt.savefig(plot_path + 'actions-scatter-plot' + plot_file_extension)
    plt.show()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=int(25.0 / dt), specifiedEpisode=42)