#%% Imports
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../')
from tensor import KoopmanTensor
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
U = np.random.rand(1,N)*action_range*np.random.choice(np.array([-1,1]), size=(1,N))
X0 = np.random.rand(2,N)*state_range*np.random.choice(np.array([-1,1]), size=(2,N))

#%% Construct snapshots of states following dynamics f
Y = f(X0, U)

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X0,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(1),
    regressor='sindy'
)

#%% Define cost function
# def cost(x, u):
#     return x.T @ Q @ x + u.T @ R @ u

def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    mat = np.vstack(np.diag(x.T @ Q @ x)) + np.power(u, 2)*R
    return mat

#%% Discretize all controls
u_bounds = np.array([[-action_range, action_range]])
step_size = 0.1
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1], step=step_size).reshape(1,-1)
All_U = np.round(All_U, decimals=1)
# All_U = U.reshape(1,-1) # continuous case is just original domain

#%% Learn control
algos = algorithmsv2.algos(
    X0,
    All_U,
    u_bounds[0],
    tensor.phi,
    tensor.psi,
    tensor.K,
    cost,
    gamma=0.5,
    epsilon=0.01,
    bellman_error_type=0,
    u_batch_size=32,
    learning_rate=1e-1,
    weight_regularization_bool=True,
    weight_regularization_lambda=0.6,
    optimizer='adam'
)
# algos.w = np.load('bellman-weights.npy')
print("Weights before updating:", algos.w)
bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
print("Weights after updating:", algos.w)

plt.plot(bellmanErrors)
plt.show()
plt.plot(gradientNorms)
plt.show()

# UPDATE (beta = 0.5)
# Mean cost 149.70096846344427, weights = 1
# 159.73050635706545, BE = 227313.64619706973
# 150.87148630617315, BE = 27789.43904288947
# 151.03942785346644, BE = 25334.162539428587
# 150.95426211613199, BE = 24971.944982697343

# Mean optimal cost 73.55472996049053

# BETA = 0.5
# 144.71, BE = 5193061.96
# 159.30, BE = 97900.11
# 159.49, BE = 9741.73
# 159.36, BE = 6387.96
# 158.92, BE = 5645.05
# 158.55, BE = 5466.73
# 157.66, BE = 4981.88
# 157.39, BE = 4845.50
# 157.15, BE = 4742.25
# 156.22, BE = 4307.26
# 154.51, BE = 3791.99
# 152.55, BE = 3328.96
# algos.w = np.array([
#     [ 0.75538592],
#     [ 0.68470443],
#     [ 0.66054487],
#     [ 1.13714188],
#     [-0.0029041 ],
#     [ 1.07005579]
# ])

# w/ discounting and weights = 1: 69.40
# w/ discounting and weights are s.t. BE = 5466.73: 69.39

# BETA = 1
# 126.80, BE = 4638908.76
# 140.57, BE = 116866.17
# 140.34, BE = 29099.91
# 139.81, BE = 7947.24
# 139.23, BE = 6387.74
# algos.w = np.array([
#     [ 1.        ]
#     [ 0.91458017]
#     [ 0.90113685]
#     [ 1.26895151]
#     [-0.09738594]
#     [ 1.13607786]
# ])

# w/o discounting: 70.03 is optimal
# w/ discounting: 68.38 is optimal

#%% Reset seed and compute initial x0s
np.random.seed(123)

num_episodes = 100
num_steps_per_episode = 100
initial_Xs = np.random.rand(2,num_episodes)*state_range*np.random.choice(np.array([-1,1]), size=(2,num_episodes)) # random initial states

#%% Construct policy
All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis = algos.pis(x)[:,0]
    # Select action column at index sampled from policy distribution
    u = np.vstack(
        All_U[:,np.random.choice(All_U_range, p=pis)]
    )
    return u

def policy2(x):
    return -C @ x

#%% Test policy by simulating system
lamb = 1e-2
costs = np.empty((num_episodes))
for episode in range(num_episodes):
    x = np.vstack(initial_Xs[:,episode])
    # print("Initial x:", x)
    cost_sum = 0
    for step in range(num_steps_per_episode):
        u = policy(x)
        # u = np.random.rand(1,1)*action_range*np.random.choice(np.array([-1,1])) # sample random action
        x_prime = f(x, u)

        # pis = algos.pis(x)[:,0]
        # (beta**step)*
        cost_sum += cost(x, u) #+ lamb * np.log(pis[All_U[0] == u[0,0]])

        x = x_prime
        # if step%250 == 0:
        #     print("Current x:", x)
    costs[episode] = cost_sum
print("Mean cost per episode:", np.mean(costs)) # Cost should be minimized

#%%