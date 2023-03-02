#%% Imports
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
    action_minimums,
    action_maximums,
    action_order,
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

#%% Construct datasets
num_episodes = 1000
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

#%% Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, size=[state_dim, N])
U = np.zeros((action_dim, N))
U = np.random.uniform(action_minimums, action_maximums, size=[action_dim, N])
Y = np.zeros_like(X)
for i in range(N):
    x = np.vstack(X[:, i])
    u = np.vstack(U[:, i])
    Y[:, i] = f(x, u)[:, 0]

#%% Estimate Koopman tensor
shotgun_tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
    # regressor='sindy'
)

#%% Save Koopman tensor
with open('./analysis/tmp/shotgun_tensor.pickle', 'wb') as handle:
    pickle.dump(shotgun_tensor, handle) #, protocol=pickle.HIGHEST_PROTOCOL)

#%% Path-based approach
X = np.zeros((num_episodes, num_steps_per_episode, state_dim))
Y = np.zeros_like(X)
U = np.zeros((num_episodes, num_steps_per_episode, action_dim))

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
).T

for episode_num in range(num_episodes):
    state = np.vstack(initial_states[episode_num])

    for step_num in range(num_steps_per_episode):
        X[episode_num, step_num] = state[:, 0]
        # action = zero_policy(state)
        action = random_policy(state)
        U[episode_num, step_num] = action[:, 0]
        state = f(state, action)
        Y[episode_num, step_num] = state[:, 0]

#%% Estimate Koopman tensor
X = X.reshape(num_episodes*num_steps_per_episode, state_dim).T
Y = Y.reshape(num_episodes*num_steps_per_episode, state_dim).T
U = U.reshape(num_episodes*num_steps_per_episode, action_dim).T

path_based_tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
    # regressor='sindy'
)

#%% Save Koopman tensor
with open('./analysis/tmp/path_based_tensor.pickle', 'wb') as handle:
    pickle.dump(path_based_tensor, handle) #, protocol=pickle.HIGHEST_PROTOCOL)

#%% Regular regression
appended_Phi_X = path_based_tensor.phi(np.concatenate((X, U), axis=0))
appended_Phi_Y = path_based_tensor.phi(np.concatenate((Y, U), axis=0))
koopman_operator = np.linalg.lstsq(appended_Phi_X.T, appended_Phi_Y.T, rcond=None)
koopman_operator = koopman_operator[0].T

#%% Save Koopman operator
with open('./analysis/tmp/extended_operator.pickle', 'wb') as handle:
    pickle.dump(koopman_operator, handle) #, protocol=pickle.HIGHEST_PROTOCOL)