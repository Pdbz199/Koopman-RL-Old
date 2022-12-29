import numpy as np
import pickle
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from dynamics import (
    action_dim,
    action_order,
    dt,
    f,
    random_policy,
    state_dim,
    state_minimums,
    state_maximums,
    state_order,
    zero_policy
)

sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor

#%% Variables for loading data
folder = "./analysis/tmp"

#%% Generate shotgun-based dataset
# num_initial_training_states = 100_000
# initial_states = np.random.uniform(
#     state_minimums,
#     state_maximums,
#     [state_dim, num_initial_training_states]
# ).T
# X = np.zeros((state_dim,num_initial_training_states))
# Y = np.zeros((state_dim,num_initial_training_states))
# U = np.zeros((action_dim,num_initial_training_states))

# for initial_training_state_num in range(num_initial_training_states):
#     initial_state = np.vstack(initial_states[initial_training_state_num])
#     X[:,initial_training_state_num] = initial_state[:,0]
#     action = zero_policy()
#     U[:,initial_training_state_num] = action[:,0]
#     Y[:,initial_training_state_num] = f(initial_state, action)[:,0]

# %% Construct Koopman tensor
# tensor = KoopmanTensor(
#     X,
#     Y,
#     U,
#     phi=observables.monomials(state_order),
#     psi=observables.monomials(action_order),
#     regressor='ols'
# )
# with open(f'{folder}/shotgun-tensor.pickle', 'wb') as handle:
#     pickle.dump(tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{folder}/shotgun-tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

#%% Test Koopman Tensor
path_length = int(10.0 / dt)
states = np.zeros((path_length,tensor.x_dim))
predicted_state_primes = np.zeros_like(states)
true_state_primes = np.zeros_like(states)
observables = np.zeros((path_length,tensor.phi_dim))
predicted_observable_primes = np.zeros_like(observables)
true_observable_primes = np.zeros_like(observables)

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    (state_dim, 1)
).T
state = np.vstack(initial_states[0])

for step_num in range(path_length):
    states[step_num] = state[:,0]
    observables[step_num] = tensor.phi(state)[:,0]

    action = zero_policy(state)
    # action = random_policy(state)

    predicted_state_primes[step_num] = tensor.f(state, action)[:,0]
    predicted_observable_primes[step_num] = tensor.phi_f(state, action)[:,0]

    state = f(state, action)

    true_state_primes[step_num] = state[:,0]
    true_observable_primes[step_num] = tensor.phi(state)[:,0]

average_state_norm = np.mean(np.linalg.norm(states, axis=0))
print(
    "Norm of difference in states (normalized by average state norm):",
    np.linalg.norm(true_state_primes - predicted_state_primes) / average_state_norm
)

average_observable_norm = np.mean(np.linalg.norm(observables, axis=0))
print(
    "Norm of difference in observables (normalized by average observable norm):",
    np.linalg.norm(true_observable_primes - predicted_observable_primes) / average_observable_norm
)