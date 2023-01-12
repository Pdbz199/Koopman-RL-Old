#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from cost import cost
from dynamics import (
    f,
    phi_dim,
    random_policy,
    state_dim,
    state_maximums,
    state_minimums,
    zero_policy
)

# Add to path to find final module
sys.path.append('../../../')

#%% Load Koopman tensors
with open('./analysis/tmp/shotgun_tensor.pickle', 'rb') as handle:
    shotgun_tensor = pickle.load(handle)

with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    path_based_tensor = pickle.load(handle)

with open('./analysis/tmp/extended_operator.pickle', 'rb') as handle:
    koopman_operator = pickle.load(handle)

#%% Generate paths
num_episodes = 100
num_steps_per_episode = 200

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
).T

true_states = np.zeros((num_episodes, num_steps_per_episode, state_dim))
shotgun_based_predicted_states = np.zeros_like(true_states)
path_based_predicted_states = np.zeros_like(true_states)
extended_koopman_operator_predicted_states = np.zeros_like(true_states)

true_observables = np.zeros((num_episodes, num_steps_per_episode, phi_dim))

actions = np.zeros((num_episodes, num_steps_per_episode, state_dim))

true_costs = np.zeros((num_episodes, num_steps_per_episode))
shotgun_based_predicted_costs = np.zeros_like(true_costs)
path_based_predicted_costs = np.zeros_like(true_costs)
extended_koopman_operator_based_predicted_costs = np.zeros_like(true_costs)

shotgun_based_state_norms = np.zeros_like(true_costs)
path_based_state_norms = np.zeros_like(true_costs)
extended_koopman_operator_based_state_norms = np.zeros_like(true_costs)

shotgun_based_observable_norms = np.zeros_like(true_costs)
path_based_observable_norms = np.zeros_like(true_costs)
extended_koopman_operator_based_observable_norms = np.zeros_like(true_costs)

for episode_num in range(num_episodes):
    # Extract initial state
    state = np.vstack(initial_states[episode_num])
    phi_state = shotgun_tensor.phi(state)

    # Set true and predicted states to same initial state
    shotgun_based_predicted_state = state
    path_based_predicted_state = state
    extended_koopman_operator_predicted_state = state
    true_state = state

    shotgun_based_predicted_observable = phi_state
    path_based_predicted_observable = phi_state
    true_observable = phi_state
    true_extended_state = np.concatenate((state, np.array([[0]])), axis=0)
    true_extended_observable = path_based_tensor.phi(true_extended_state)
    extended_koopman_operator_predicted_observable = true_extended_observable

    for step_num in range(num_steps_per_episode):
        # Save states in arrays
        true_states[episode_num, step_num] = true_state[:, 0]
        true_observables[episode_num, step_num] = true_observable[:, 0]
        shotgun_based_predicted_states[episode_num, step_num] = shotgun_based_predicted_state[:, 0]
        path_based_predicted_states[episode_num, step_num] = path_based_predicted_state[:, 0]
        extended_koopman_operator_predicted_states[episode_num, step_num] = extended_koopman_operator_predicted_state[:3, 0]

        # Compute norms
        shotgun_based_state_norms[episode_num, step_num] = np.linalg.norm(true_state - shotgun_based_predicted_state)
        path_based_state_norms[episode_num, step_num] = np.linalg.norm(true_state - path_based_predicted_state)
        extended_koopman_operator_based_state_norms[episode_num, step_num] = np.linalg.norm(true_state - extended_koopman_operator_predicted_state)

        shotgun_based_observable_norms[episode_num, step_num] = np.linalg.norm(true_observable - shotgun_based_predicted_observable)
        path_based_observable_norms[episode_num, step_num] = np.linalg.norm(true_observable - path_based_predicted_observable)
        extended_koopman_operator_based_observable_norms[episode_num, step_num] = np.linalg.norm(true_extended_observable - extended_koopman_operator_predicted_observable)

        # Get action
        # action = zero_policy(true_state)
        action = random_policy(true_state)
        actions[episode_num, step_num] = action[:, 0]

        # Compute costs
        true_costs[episode_num, step_num] = cost(true_state, action)
        shotgun_based_predicted_costs[episode_num, step_num] = cost(shotgun_based_predicted_state, action)
        path_based_predicted_costs[episode_num, step_num] = cost(path_based_predicted_state, action)
        # extended_koopman_operator_based_predicted_costs[episode_num, step_num] = cost(extended_koopman_operator_predicted_state, action)

        # Update states
        shotgun_based_predicted_observable = shotgun_tensor.phi_f(true_state, action)
        shotgun_based_predicted_state = shotgun_tensor.B.T @ shotgun_based_predicted_observable

        path_based_predicted_observable = path_based_tensor.phi_f(true_state, action)
        path_based_predicted_state = path_based_tensor.B.T @ path_based_predicted_observable

        true_extended_state = np.concatenate((true_state, action), axis=0)
        true_extended_observable = path_based_tensor.phi(true_extended_state)
        extended_koopman_operator_predicted_observable = koopman_operator @ true_extended_observable
        extended_koopman_operator_predicted_state = extended_koopman_operator_predicted_observable[1:4]

        true_state = f(true_state, action)
        true_observable = shotgun_tensor.phi(true_state)

#%% Plots
fig = plt.figure()
episode_range = np.arange(num_episodes)
step_range = np.arange(num_steps_per_episode)

#%% Plot average state norm per step across episodes
shotgun_based_average_norms_per_step = shotgun_based_state_norms.mean(axis=0)
path_based_average_norms_per_step = path_based_state_norms.mean(axis=0)
extended_koopman_operator_based_average_norms_per_step = extended_koopman_operator_based_state_norms.mean(axis=0)
average_state_norm = np.linalg.norm(true_states, axis=2).mean(axis=1).mean()
print("Average state norm:", average_state_norm)

ax = fig.add_subplot(2, 1, 1)
ax.set_title(f"""
Average State Norm per Step (Normalized by Average State Norm)
Average State Norm: {average_state_norm}
""")
ax.plot(shotgun_based_average_norms_per_step / average_state_norm)
ax.plot(path_based_average_norms_per_step / average_state_norm)
ax.plot(extended_koopman_operator_based_average_norms_per_step / average_state_norm)
ax.legend(['Shotgun-based', 'Path-based', 'Extended Koopman Operator'])

#%% Plot average observable norm per step across episodes
shotgun_based_average_observable_norms_per_step = shotgun_based_observable_norms.mean(axis=0)
path_based_average_observable_norms_per_step = path_based_observable_norms.mean(axis=0)
extended_koopman_operator_based_average_observable_norms_per_step = extended_koopman_operator_based_observable_norms.mean(axis=0)
average_observable_norm = np.linalg.norm(true_observables, axis=2).mean(axis=1).mean()

ax = fig.add_subplot(2, 1, 2)
ax.set_title(f"""
Average Observable Norm per Step (Normalized by Average Observable Norm)
Average Observable Norm: {average_observable_norm}
""")
ax.plot(shotgun_based_average_observable_norms_per_step / average_observable_norm)
ax.plot(path_based_average_observable_norms_per_step / average_observable_norm)
ax.plot(extended_koopman_operator_based_average_observable_norms_per_step / average_observable_norm)
ax.legend(['Shotgun-based', 'Path-based', 'Extended Koopman Operator'])

#%% Shot plots
plt.tight_layout()
plt.show()