#%%
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% System dynamics
A = np.array([
    [0.5, 0.0],
    [0.0, 0.3]
], dtype=np.float64)
B = np.array([
    [1.0],
    [1.0]
], dtype=np.float64)
Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
], dtype=np.float64)
# R = 1
R = np.array([[1.0]], dtype=np.float64)

def f(x, u):
    return A @ x + B @ u

#%% Construct snapshots of u from random agent and initial states x0
N = 10000
state_range = 10
action_range = 10
X = np.random.rand(2,N)*state_range*np.random.choice(np.array([-1,1]), size=(2,N))
U = np.random.rand(1,N)*action_range*np.random.choice(np.array([-1,1]), size=(1,N))

#%% Construct snapshots of states following dynamics f
Y = f(X, U)

#%% Estimate Koopman tensor
ols_tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(2),
    regressor='ols',
    p_inv=False
)

sindy_tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(2),
    regressor='sindy'
)

#%% Training error
ols_norms = np.empty((N))
sindy_norms = np.empty((N))
for i in range(N):
    phi_x = np.vstack(ols_tensor.Phi_X[:,i]) # current (lifted) state

    action = np.vstack(ols_tensor.U[:,i])

    true_phi_x_prime = np.vstack(ols_tensor.Phi_Y[:, i])
    ols_predicted_phi_x_prime = ols_tensor.K_(action) @ phi_x
    sindy_predicted_phi_x_prime = sindy_tensor.K_(action) @ phi_x

    # Compute norms
    ols_norms[i] = utilities.l2_norm(true_phi_x_prime, ols_predicted_phi_x_prime)
    sindy_norms[i] = utilities.l2_norm(true_phi_x_prime, sindy_predicted_phi_x_prime)
print("Training error (OLS):", np.mean(ols_norms))
print("Training error (SINDy):", np.mean(sindy_norms))

#%% Testing error normalized by mean norm of different starting states
num_episodes = 100
num_steps_per_episode = 100

ols_norms = np.empty((num_episodes,num_steps_per_episode))
sindy_norms = np.empty((num_episodes,num_steps_per_episode))
X_sample = np.random.rand(2,num_episodes)*state_range*np.random.choice(np.array([-1,1]), size=(2,num_episodes))

#%%
for episode in range(num_episodes):
    x = np.vstack(X_sample[:,episode]) # initial state

    for step in range(num_steps_per_episode):
        phi_x = ols_tensor.phi(x)

        action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1])) # sample random action

        true_x_prime = f(x, action)
        true_phi_x_prime = ols_tensor.phi(true_x_prime)
        ols_predicted_phi_x_prime = ols_tensor.K_(action) @ phi_x
        sindy_predicted_phi_x_prime = sindy_tensor.K_(action) @ phi_x

        ols_norms[episode,step] = utilities.l2_norm(true_phi_x_prime, ols_predicted_phi_x_prime)
        sindy_norms[episode,step] = utilities.l2_norm(true_phi_x_prime, sindy_predicted_phi_x_prime)

        x = true_x_prime

print("Testing error over all episodes (OLS):", np.mean(ols_norms))
print("Testing error over all episodes (SINDy):", np.mean(sindy_norms))

#%%