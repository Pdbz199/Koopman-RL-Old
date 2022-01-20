#%% Imports
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../../')
import estimate_L
import observables
import utilities

from control import lqr

#%% System dynamics
A = np.array([
    [0.5, 0],
    [0, 0.3]
])
B = np.array([
    [1],
    [1]
])
Q = np.array([
    [1, 0],
    [0, 1]
])
R = 1

def f(x, u):
    return A @ x + B @ u

#%% Traditional LQR
lq = lqr(A, B, Q, R)
K = lq[0][0] # [ 8.19803903 12.97251466]
# lq[0] == [[ 8.19803903 12.97251466]]
# lq[1] == [[1.15898133 0.08198039]
#           [0.08198039 0.12972515]]
# lq[2] == [-9.947309 +0.j -1.0252059+0.j]

#%% Construct snapshots of u from random agent and initial states x0
N = 10000
action_range = 10
state_range = 10
U = np.random.rand(1,N)*action_range
X0 = np.random.rand(2,N)*state_range

#%% Construct snapshots of states following dynamics f
Y = f(X0, U)

#%% Estimate Koopman tensor
order = 2
phi = observables.monomials(order)
psi = observables.monomials(order)

#%% Build Phi and Psi matrices
Phi_X = phi(X0)
Phi_Y = phi(Y)
Psi_U = psi(U)
dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]
N = X0.shape[1]

print(Phi_X.shape)

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
B_ = estimate_L.ols(Phi_X.T, X0.T)

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    # return np.einsum('ijz,kz->ij', K, psi(u))
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% Training error
norms = np.empty((N))
for i in range(N):
    phi_x = np.vstack(Phi_X[:,i]) # current (lifted) state

    action = np.vstack(U[:,i])

    true_x_prime = np.vstack(Y[:,i])
    predicted_x_prime = B_.T @ K_u(K, action) @ phi_x

    # Compute norms
    norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
print("Training error:", np.mean(norms))

#%% Testing error normalized by mean norm of different starting states
num_episodes = 100
num_steps_per_episode = 100

norms = np.empty((num_episodes,N))
norms_states = np.empty((num_episodes,N))
X0_sample = np.random.rand(2,num_episodes)*state_range # random initial states
norm_X0s = utilities.l2_norm(X0_sample, np.zeros_like(X0_sample))
avg_norm_X0s = np.mean(norm_X0s)

for episode in range(num_episodes):
    x = np.vstack(X0_sample[:,episode])
    phi_x = phi(x) # apply phi to initial state

    for step in range(num_steps_per_episode):
        action = np.random.rand(1,1)*action_range # sample random action

        true_x_prime = f(x, action)
        predicted_x_prime = B_.T @ K_u(K, action) @ phi_x

        norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
        norms_states[episode,step] = utilities.l2_norm(x, np.zeros_like(x))

        x = true_x_prime
avg_norm_by_path = np.mean(norms_states, axis = 1)
print("Avg testing error over all episodes:", np.mean(norms))
print("Avg testing error over all episodes normalized by avg norm of starting state:", np.mean(norms)/avg_norm_X0s)
print("Avg testing error over all episodes normalized by avg norm of state path:", np.mean((np.mean(norms, axis =1)/avg_norm_by_path)))

#%%