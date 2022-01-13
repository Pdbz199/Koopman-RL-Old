#%% Imports
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../../')
# import algorithmsv2
import algorithmsv2_parallel as algorithmsv2
import estimate_L
import observables
import utilities

# from control import lqr

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
# R = np.array([[1]])

def f(x, u):
    return A @ x + B @ u

#%% Traditional LQR
# lq = lqr(A, B, Q, R)
# C = lq[0][0]
# lq[0] == [[ 8.96688317 -6.28428936]]
# lq[1] == [[ 79.40499383 -70.43811066]
#           [-70.43811066  64.1538213 ]]
# lq[2] == [-1.474176 +0.j -0.4084178+0.j]

#%% Construct snapshots of u from random agent and initial states x0
N = 10000
action_range = 10
state_range = 10 #! Are these reasonable ranges? Depending on draws before we were setting the seed properly, we were seeing better/worse convergence of ABE with different learning rates.
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
def cost(x, u):
    return x.T @ Q @ x + np.power(u, 2) * R # u.T @ R @ u

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
# algos.w = np.load('bellman-weights.npy')
algos.w = np.array([
    [ 1.        ],
    [ 0.04950633],
    [-0.32755624],
    [ 1.32549688],
    [ 0.01576367],
    [ 1.11493135]
])
# algos.w = np.array([
#     [ 1.        ],
#     [ 4.90417142],
#     [ 4.04355341],
#     [67.17531854],
#     [67.25611674],
#     [61.07985195]
# ])
print("Weights before updating:", algos.w)
bellmanErrors, gradientNorms = algos.algorithm2(batch_size=64)
print("Weights after updating:", algos.w)

#%% Reset seed
np.random.seed(123)

#%% Retrieve policy
def policy(x):
    pis = algos.pis(x)[:,0]
    # Select action column at index sampled from policy distribution
    u = np.vstack(
        All_U[:,np.random.choice(np.arange(All_U.shape[1]), p=pis)]
    )
    return u

#%% Test policy by simulating system
num_episodes = 100
num_steps_per_episode = 100
costs = np.empty((num_episodes))
for episode in range(num_episodes):
    x = np.vstack(
        np.random.rand(2,1)*state_range # random initial state
    )
    print("Initial x:", x)
    cost_sum = 0
    for step in range(num_steps_per_episode):
        u = policy(x)
        # u = np.random.rand(1,1)*action_range # sample random action
        x_prime = f(x, u)

        cost_sum += cost(x, u)

        x = x_prime
        if not step%250:
            print("Current x:", x)
    costs[episode] = cost_sum
print("Mean cost per episode:", np.mean(costs)) # Cost should be minimized

#%%