#%% Imports
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
# import algorithmsv2
import algorithmsv2_parallel as algorithmsv2
import observables
import utilities

from control import dlqr

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

#%% Traditional LQR
lq = dlqr(A, B, Q, R)
C = lq[0]
# lq[0] == [[ 8.96688317 -6.28428936]]
# lq[1] == [[ 79.40499383 -70.43811066]
#           [-70.43811066  64.1538213 ]]
# lq[2] == [-1.474176 +0.j -0.4084178+0.j]

#%% Construct snapshots of u from random agent and initial states x0
N = 10000
action_range = 10
state_range = 10
U = np.random.rand(1,N)*action_range*np.random.choice(np.array([-1,1]), size=(1,N))
X0 = np.random.rand(2,N)*state_range*np.random.choice(np.array([-1,1]), size=(2,N))

#%% Construct snapshots of states following dynamics f
Y = f(X0, U)

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X0,
    Y,
    U,
    phi=observables.monomials(1),
    psi=observables.monomials(1),
    regressor='sindy'
)

#%% Training error
norms = np.empty((N))
for i in range(N):
    phi_x = np.vstack(tensor.Phi_X[:,i]) # current (lifted) state

    action = np.vstack(tensor.U[:,i])

    # true_x_prime = np.vstack(tensor.Y[:,i])
    true_phi_x_prime = np.vstack(tensor.Phi_Y[:,i])
    predicted_phi_x_prime = tensor.K_(action) @ phi_x

    # Compute norms
    norms[i] = utilities.l2_norm(true_phi_x_prime, predicted_phi_x_prime)
print("Training error:", np.mean(norms))

#%% Testing error normalized by mean norm of different starting states
num_episodes = 100
num_steps_per_episode = 100

norms = np.empty((num_episodes,num_steps_per_episode))
norms_states = np.empty((num_episodes,num_steps_per_episode))
X0_sample = np.random.rand(2,num_episodes)*state_range*np.random.choice(np.array([-1,1]), size=(2,num_episodes)) # random initial states
norm_X0s = utilities.l2_norm(X0_sample, np.zeros_like(X0_sample))
avg_norm_X0s = np.mean(norm_X0s)

for episode in range(num_episodes):
    x = np.vstack(X0_sample[:,episode]) # initial state

    for step in range(num_steps_per_episode):
        phi_x = tensor.phi(x) # apply phi to state

        action = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1])) # sample random action

        true_x_prime = f(x, action)
        true_phi_x_prime = tensor.phi(true_x_prime)
        predicted_phi_x_prime = tensor.K_(action) @ phi_x

        norms[episode,step] = utilities.l2_norm(true_phi_x_prime, predicted_phi_x_prime)
        norms_states[episode,step] = utilities.l2_norm(x, np.zeros_like(x))

        x = true_x_prime

avg_norm_by_path = np.mean(norms_states, axis = 1)
print("Avg testing error over all episodes:", np.mean(norms))
print("Avg testing error over all episodes normalized by avg norm of starting state:", np.mean(norms)/avg_norm_X0s)
print("Avg testing error over all episodes normalized by avg norm of state path:", np.mean((np.mean(norms, axis =1)/avg_norm_by_path)))

#%%