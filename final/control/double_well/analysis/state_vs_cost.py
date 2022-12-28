import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from cost import cost, Q, R, reference_point
from dynamics import (
    action_dim,
    all_actions,
    action_order,
    continuous_A,
    continuous_B,
    continuous_f,
    dt,
    f,
    state_column_shape,
    state_dim,
    state_minimums,
    state_maximums,
    state_order,
    zero_policy
)
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.integrate import solve_ivp

sys.path.append('../../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from final.control.policies.lqr import LQRPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0
# reg_lambda = 0.1

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

# Koopman discrete actor critic policy
actor_critic_policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    '../saved_models/double-well-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.003,
    load_model=True
)

# Koopman value iteration policy
value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    '../saved_models/double-well-discrete-value-iteration-policy.pt',
    dt=dt,
    seed=seed
)

#%% Get set of initial states
num_initial_states_to_generate = 1000
initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_initial_states_to_generate]
).T

#%% Set number of steps per path
num_steps_per_path = int(10.0 / dt)

# #%% Create arrays for tracking states, actions, and costs
# actor_critic_states = np.empty((num_initial_states_to_generate, num_steps_per_path, state_dim))
# actor_critic_actions = np.empty((num_initial_states_to_generate, num_steps_per_path, action_dim))
# actor_critic_costs = np.empty((num_initial_states_to_generate, num_steps_per_path))
# value_iteration_states = np.empty_like(actor_critic_states)
# value_iteration_actions = np.empty_like(actor_critic_actions)
# value_iteration_costs = np.empty_like(actor_critic_costs)
# lqr_states = np.empty_like(actor_critic_states)
# lqr_actions = np.empty_like(actor_critic_actions)
# lqr_costs = np.empty_like(actor_critic_costs)

# #%% Generate a path for each initial state
# for episode_num in range(num_initial_states_to_generate):
#     # Get initial state from pre-generated list
#     initial_state = initial_states[episode_num]

#     # Initialize states
#     state = np.vstack(initial_state) # column vector
#     actor_critic_state = state
#     value_iteration_state = state
#     lqr_state = state

#     # Generate steps along the path
#     for step_num in range(num_steps_per_path):
#         # Save latest states
#         actor_critic_states[episode_num, step_num] = state[:,0]
#         value_iteration_states[episode_num, step_num] = state[:,0]
#         lqr_states[episode_num, step_num] = state[:,0]

#         # Get actions for current states
#         actor_critic_action, _ = actor_critic_policy.get_action(state)
#         value_iteration_action = value_iteration_policy.get_action(state)
#         lqr_action = lqr_policy.get_action(lqr_state)

#         # Save actions
#         actor_critic_actions[episode_num, step_num] = actor_critic_action
#         value_iteration_actions[episode_num, step_num] = value_iteration_action
#         lqr_actions[episode_num, step_num] = lqr_action

#         # Compute costs of state-action pairs
#         actor_critic_costs[episode_num, step_num] = cost(actor_critic_state, actor_critic_action)
#         value_iteration_costs[episode_num, step_num] = cost(value_iteration_state, value_iteration_action)
#         lqr_costs[episode_num, step_num] = cost(lqr_state, lqr_action)

#         # Update the states
#         actor_critic_state = f(actor_critic_state, actor_critic_action)
#         value_iteration_state = f(value_iteration_state, value_iteration_action)
#         lqr_state = f(lqr_state, lqr_action)

#%% Set variables for saving arrays
tmp_dir = './tmp'
array_types = ['states', 'actions', 'costs']
controller_types = ['actor_critic', 'value_iteration', 'lqr']
extension = 'npy'

#%% Save numpy arrays just in case
# np.save(f"{tmp_dir}/{controller_types[0]}_{array_types[0]}.{extension}", actor_critic_states)
# np.save(f"{tmp_dir}/{controller_types[0]}_{array_types[1]}.{extension}", actor_critic_actions)
# np.save(f"{tmp_dir}/{controller_types[0]}_{array_types[2]}.{extension}", actor_critic_costs)
# np.save(f"{tmp_dir}/{controller_types[1]}_{array_types[0]}.{extension}", value_iteration_states)
# np.save(f"{tmp_dir}/{controller_types[1]}_{array_types[1]}.{extension}", value_iteration_actions)
# np.save(f"{tmp_dir}/{controller_types[1]}_{array_types[2]}.{extension}", value_iteration_costs)
# np.save(f"{tmp_dir}/{controller_types[2]}_{array_types[0]}.{extension}", lqr_states)
# np.save(f"{tmp_dir}/{controller_types[2]}_{array_types[1]}.{extension}", lqr_actions)
# np.save(f"{tmp_dir}/{controller_types[2]}_{array_types[2]}.{extension}", lqr_costs)

#%% Load numpy arrays
actor_critic_states = np.load(f"{tmp_dir}/{controller_types[0]}_{array_types[0]}.{extension}")
actor_critic_actions = np.load(f"{tmp_dir}/{controller_types[0]}_{array_types[1]}.{extension}")
actor_critic_costs = np.load(f"{tmp_dir}/{controller_types[0]}_{array_types[2]}.{extension}")
value_iteration_states = np.load(f"{tmp_dir}/{controller_types[1]}_{array_types[0]}.{extension}")
value_iteration_actions = np.load(f"{tmp_dir}/{controller_types[1]}_{array_types[1]}.{extension}")
value_iteration_costs = np.load(f"{tmp_dir}/{controller_types[1]}_{array_types[2]}.{extension}")
lqr_states = np.load(f"{tmp_dir}/{controller_types[2]}_{array_types[0]}.{extension}")
lqr_actions = np.load(f"{tmp_dir}/{controller_types[2]}_{array_types[1]}.{extension}")
lqr_costs = np.load(f"{tmp_dir}/{controller_types[2]}_{array_types[2]}.{extension}")

#%% Create figure
fig = plt.figure()

#%% Choose episode number to plot
# Picked final episode for demonstration
episode_number_to_plot = num_initial_states_to_generate - 1

#%% Plot individual state dimensions over time
ax = fig.add_subplot(311)
ax.set_title("States over time")
ax.plot(actor_critic_states[episode_number_to_plot, :, 0])
ax.plot(actor_critic_states[episode_number_to_plot, :, 1])
ax.legend(['x_0', 'x_1'])

#%% Plot actions over time
ax = fig.add_subplot(312)
ax.plot(actor_critic_actions[episode_number_to_plot, :, 0])

#%% Plot costs over time
ax = fig.add_subplot(313)
ax.plot(actor_critic_costs[episode_number_to_plot])

#%% Show plots
plt.show()

#%% Compute average cost for each initial state
actor_critic_average_cost_per_initial_state = actor_critic_costs.mean(axis=0)
value_iteration_average_cost_per_initial_state = value_iteration_costs.mean(axis=0)
lqr_average_cost_per_initial_state = lqr_costs.mean(axis=0)

#%% Plot costs vs initial state
fig = plt.figure()

#%% Plot actor-critic policy costs vs initial states
ax = fig.add_subplot(131, projection='3d')
ax.scatter(actor_critic_states[:, 0, 0], actor_critic_states[:, 0, 1], actor_critic_average_cost_per_initial_state)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("cost")
plt.savefig(f"{tmp_dir}/{controller_types[0]}_costs_vs_initial_state.svg")
plt.savefig(f"{tmp_dir}/{controller_types[0]}_costs_vs_initial_state.png")

#%% Plot value iteration policy costs vs initial states
ax = fig.add_subplot(132, projection='3d')
ax.scatter(value_iteration_states[:, 0, 0], value_iteration_states[:, 0, 1], value_iteration_average_cost_per_initial_state)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("cost")
plt.savefig(f"{tmp_dir}/{controller_types[1]}_costs_vs_initial_state.svg")
plt.savefig(f"{tmp_dir}/{controller_types[1]}_costs_vs_initial_state.png")

#%% Plot LQR costs vs initial states
ax = fig.add_subplot(133, projection='3d')
ax.scatter(lqr_states[:, 0, 0], lqr_states[:, 0, 1], lqr_average_cost_per_initial_state)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("cost")
plt.savefig(f"{tmp_dir}/{controller_types[2]}_costs_vs_initial_state.svg")
plt.savefig(f"{tmp_dir}/{controller_types[2]}_costs_vs_initial_state.png")

#%% Show plots
plt.show()