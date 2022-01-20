#%% Imports
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../../')
# import algorithmsv2
import algorithmsv2_parallel as algorithmsv2
import estimate_L
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

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    # return np.einsum('ijz,kz->ij', K, psi(u))
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% Define cost function
# def cost(x, u):
#     return x.T @ Q @ x + u.T @ R @ u

def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    mat = np.vstack(np.diag(x.T @ Q @ x)) + np.power(u, 2)*R
    return mat

#%% Discretize all controls
u_bounds = np.array([[0.0, action_range]])
step_size = 0.1
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1], step=step_size).reshape(1,-1)
# All_U = U.reshape(1,-1) # continuous case is just original domain

#%% Learn control
epsilon = 1 # 0.1
algos = algorithmsv2.algos(
    X0,
    All_U,
    u_bounds[0],
    phi,
    psi,
    K,
    cost,
    epsilon=epsilon,
    bellmanErrorType=0,
    weightRegularizationBool=0,
    u_batch_size=30,
    learning_rate=1e-4
)
# algos.w = np.array([
#     [ 1.        ],
#     [-0.10849056],
#     [-0.3513768 ],
#     [ 1.33814896],
#     [ 0.0154977 ],
#     [ 1.11888816]
# ])
# algos.w = np.load('bellman-weights.npy')
print("Weights before updating:", algos.w)
# bellmanErrors, gradientNorms = algos.algorithm2(batch_size=256)
# print("Weights after updating:", algos.w)

#%% Reset seed and compute initial x0s
np.random.seed(123)

num_episodes = 100
num_steps_per_episode = 100
initial_Xs = np.random.rand(2,num_episodes)*state_range # random initial states

#%% Construct policy
All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis = algos.pis(x)[:,0]
    # Select action column at index sampled from policy distribution
    u = np.vstack(
        All_U[:,np.random.choice(All_U_range, p=pis)]
    )
    return u

# def policy(x):
#     return -C @ x

#%% Test policy by simulating system
costs = np.empty((num_episodes))
for episode in range(num_episodes):
    x = np.vstack(initial_Xs[:,episode])
    # print("Initial x:", x)
    cost_sum = 0
    for step in range(num_steps_per_episode):
        u = policy(x)
        # u = np.random.rand(1,1)*action_range # sample random action
        x_prime = f(x, u)

        cost_sum += cost(x, u)

        x = x_prime
        # if step%250 == 0:
        #     print("Current x:", x)
    costs[episode] = cost_sum
print("Mean cost per episode:", np.mean(costs)) # Cost should be minimized

#%%