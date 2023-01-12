import numpy as np
import pickle
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
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

sys.path.append('../../../')
# import final.observables as observables
# from final.tensor import KoopmanTensor
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from final.control.policies.lqr import LQRPolicy

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

with open(f'./analysis/tmp/shotgun-tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

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
    'saved_models/double-well-discrete-actor-critic-policy.pt',
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
    'saved_models/double-well-discrete-value-iteration-policy.pt',
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
num_episodes = 5
num_steps_per_path = int(10.0 / dt)

#%% Set variables for saving arrays
array_types = ['states', 'actions', 'costs']
controller_types = ['actor_critic', 'value_iteration', 'lqr']
extension = 'npy'

#%% Function for loading arrays
def load_arrays(folder="./analysis/tmp", prefix="", suffix=""):
    np_actor_critic_states = np.load(f"{folder}/{prefix}{controller_types[0]}_{array_types[0]}{suffix}.{extension}")
    np_actor_critic_actions = np.load(f"{folder}/{prefix}{controller_types[0]}_{array_types[1]}{suffix}.{extension}")
    np_actor_critic_costs = np.load(f"{folder}/{prefix}{controller_types[0]}_{array_types[2]}{suffix}.{extension}")
    np_value_iteration_states = np.load(f"{folder}/{prefix}{controller_types[1]}_{array_types[0]}{suffix}.{extension}")
    np_value_iteration_actions = np.load(f"{folder}/{prefix}{controller_types[1]}_{array_types[1]}{suffix}.{extension}")
    np_value_iteration_costs = np.load(f"{folder}/{prefix}{controller_types[1]}_{array_types[2]}{suffix}.{extension}")
    np_lqr_states = np.load(f"{folder}/{prefix}{controller_types[2]}_{array_types[0]}{suffix}.{extension}")
    np_lqr_actions = np.load(f"{folder}/{prefix}{controller_types[2]}_{array_types[1]}{suffix}.{extension}")
    np_lqr_costs = np.load(f"{folder}/{prefix}{controller_types[2]}_{array_types[2]}{suffix}.{extension}")

    return (
        np_actor_critic_states,
        np_actor_critic_actions,
        np_actor_critic_costs,
        np_value_iteration_states,
        np_value_iteration_actions,
        np_value_iteration_costs,
        np_lqr_states,
        np_lqr_actions,
        np_lqr_costs
    )

def main():
    #%% Import pymp
    import pymp

    #%% Create arrays for tracking states, actions, and costs
    actor_critic_states = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path, state_dim))
    actor_critic_actions = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path, action_dim))
    actor_critic_costs = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path))
    value_iteration_states = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path, state_dim))
    value_iteration_actions = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path, action_dim))
    value_iteration_costs = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path))
    lqr_states = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path, state_dim))
    lqr_actions = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path, action_dim))
    lqr_costs = pymp.shared.array((num_initial_states_to_generate, num_episodes, num_steps_per_path))

    #%% Function to save numpy arrays
    def save_arrays(folder="./analysis/tmp", prefix="", suffix=""):
        np_actor_critic_states = np.array(actor_critic_states)
        np_actor_critic_actions = np.array(actor_critic_actions)
        np_actor_critic_costs = np.array(actor_critic_costs)
        np_value_iteration_states = np.array(value_iteration_states)
        np_value_iteration_actions = np.array(value_iteration_actions)
        np_value_iteration_costs = np.array(value_iteration_costs)
        np_lqr_states = np.array(lqr_states)
        np_lqr_actions = np.array(lqr_actions)
        np_lqr_costs = np.array(lqr_costs)

        np.save(f"{folder}/{prefix}{controller_types[0]}_{array_types[0]}{suffix}.{extension}", np_actor_critic_states)
        np.save(f"{folder}/{prefix}{controller_types[0]}_{array_types[1]}{suffix}.{extension}", np_actor_critic_actions)
        np.save(f"{folder}/{prefix}{controller_types[0]}_{array_types[2]}{suffix}.{extension}", np_actor_critic_costs)
        np.save(f"{folder}/{prefix}{controller_types[1]}_{array_types[0]}{suffix}.{extension}", np_value_iteration_states)
        np.save(f"{folder}/{prefix}{controller_types[1]}_{array_types[1]}{suffix}.{extension}", np_value_iteration_actions)
        np.save(f"{folder}/{prefix}{controller_types[1]}_{array_types[2]}{suffix}.{extension}", np_value_iteration_costs)
        np.save(f"{folder}/{prefix}{controller_types[2]}_{array_types[0]}{suffix}.{extension}", np_lqr_states)
        np.save(f"{folder}/{prefix}{controller_types[2]}_{array_types[1]}{suffix}.{extension}", np_lqr_actions)
        np.save(f"{folder}/{prefix}{controller_types[2]}_{array_types[2]}{suffix}.{extension}", np_lqr_costs)

    #%% Allow for nested parallel loops
    pymp.config.nested = True

    #%% Initialize parallel path counter
    # counter = pymp.shared.array((1), dtype='uint8')

    #%% Thread counts if parallelizing paths and episodes
    num_path_threads = 4
    num_episode_threads = 4

    #%% Thread counts if parallelizing paths
    # num_path_threads = 8

    #%% Generate paths for each initial state
    with pymp.Parallel(num_path_threads) as p1:
        with pymp.Parallel(num_episode_threads) as p2:
            for path_num in p1.range(0, num_initial_states_to_generate):
            # for path_num in range(num_initial_states_to_generate):

                # Get initial state from pre-generated list
                initial_state = initial_states[path_num]

                for episode_num in p2.range(0, num_episodes):
                # for episode_num in range(num_episodes):
                    # Initialize states
                    state = np.vstack(initial_state) # column vector
                    actor_critic_state = state
                    value_iteration_state = state
                    lqr_state = state

                    # Generate steps along the path
                    for step_num in range(num_steps_per_path):
                        # Save latest states
                        actor_critic_states[path_num, episode_num, step_num] = actor_critic_state[:,0]
                        value_iteration_states[path_num, episode_num, step_num] = value_iteration_state[:,0]
                        lqr_states[path_num, episode_num, step_num] = lqr_state[:,0]

                        # Get actions for current states
                        actor_critic_action, _ = actor_critic_policy.get_action(actor_critic_state)
                        value_iteration_action = value_iteration_policy.get_action(value_iteration_state)
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

            save_arrays(folder="./analysis/tmp/test", prefix="TEST-")

if __name__ == '__main__':
    main()