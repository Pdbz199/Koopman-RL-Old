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
    dt,
    f,
    state_dim,
    state_minimums,
    state_maximums,
    state_order
)

sys.path.append('../../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from final.control.policies.lqr import LQRPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0

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
num_episodes = 1000
num_steps_per_path = int(10.0 / dt)

# #%% Create arrays for tracking states, actions, and costs
actor_critic_states = np.empty((num_initial_states_to_generate, num_episodes, num_steps_per_path, state_dim))
actor_critic_actions = np.empty((num_initial_states_to_generate, num_episodes, num_steps_per_path, action_dim))
actor_critic_costs = np.empty((num_initial_states_to_generate, num_episodes, num_steps_per_path))
value_iteration_states = np.empty_like(actor_critic_states)
value_iteration_actions = np.empty_like(actor_critic_actions)
value_iteration_costs = np.empty_like(actor_critic_costs)
lqr_states = np.empty_like(actor_critic_states)
lqr_actions = np.empty_like(actor_critic_actions)
lqr_costs = np.empty_like(actor_critic_costs)

#%% Generate paths for each initial state
for path_num in range(num_initial_states_to_generate):
    # Get initial state from pre-generated list
    initial_state = initial_states[path_num]

    for episode_num in range(num_episodes):
        # Initialize states
        state = np.vstack(initial_state) # column vector
        actor_critic_state = state
        value_iteration_state = state
        lqr_state = state

        # Generate steps along the path
        for step_num in range(num_steps_per_path):
            # Save latest states
            actor_critic_states[path_num, episode_num, step_num] = state[:,0]
            value_iteration_states[path_num, episode_num, step_num] = state[:,0]
            lqr_states[path_num, episode_num, step_num] = state[:,0]

            # Get actions for current states
            actor_critic_action, _ = actor_critic_policy.get_action(state)
            value_iteration_action = value_iteration_policy.get_action(state)
            lqr_action = lqr_policy.get_action(lqr_state)

            # Save actions
            actor_critic_actions[path_num, episode_num, step_num] = actor_critic_action
            value_iteration_actions[path_num, episode_num, step_num] = value_iteration_action
            lqr_actions[path_num, episode_num, step_num] = lqr_action

            # Compute costs of state-action pairs
            actor_critic_costs[path_num, episode_num, step_num] = cost(actor_critic_state, actor_critic_action)
            value_iteration_costs[path_num, episode_num, step_num] = cost(value_iteration_state, value_iteration_action)
            lqr_costs[path_num, episode_num, step_num] = cost(lqr_state, lqr_action)

            # Update the states
            actor_critic_state = f(actor_critic_state, actor_critic_action)
            value_iteration_state = f(value_iteration_state, value_iteration_action)
            lqr_state = f(lqr_state, lqr_action)

#%% Set variables for saving arrays
tmp_dir = './tmp'
array_types = ['states', 'actions', 'costs']
controller_types = ['actor_critic', 'value_iteration', 'lqr']
extension = 'npy'

#%% Save numpy arrays
np.save(f"{tmp_dir}/{controller_types[0]}_{array_types[0]}.{extension}", actor_critic_states)
np.save(f"{tmp_dir}/{controller_types[0]}_{array_types[1]}.{extension}", actor_critic_actions)
np.save(f"{tmp_dir}/{controller_types[0]}_{array_types[2]}.{extension}", actor_critic_costs)
np.save(f"{tmp_dir}/{controller_types[1]}_{array_types[0]}.{extension}", value_iteration_states)
np.save(f"{tmp_dir}/{controller_types[1]}_{array_types[1]}.{extension}", value_iteration_actions)
np.save(f"{tmp_dir}/{controller_types[1]}_{array_types[2]}.{extension}", value_iteration_costs)
np.save(f"{tmp_dir}/{controller_types[2]}_{array_types[0]}.{extension}", lqr_states)
np.save(f"{tmp_dir}/{controller_types[2]}_{array_types[1]}.{extension}", lqr_actions)
np.save(f"{tmp_dir}/{controller_types[2]}_{array_types[2]}.{extension}", lqr_costs)