#%% Imports
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
# import algorithmsv2
import algorithmsv2_parallel as algorithmsv2
import observables

from control import dlqr, dare

#%% System dynamics
gamma = 0.5
lamb = 1.0 # 0.6
tau = 0.002

#%% True System Dynamics
m = 1 # pendulum mass
M = 5 # cart mass
L = 2 # pendulum length
g = -10 # gravitational acceleration
d = 1 # (delta) cart damping

b = 1 # pendulum up (b = 1)
w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])

A = tau * np.array([
    [1 / tau, 1, 0, 0],
    [0, 1 / tau - d / M, b * m * g / M, 0],
    [0, 0, 1/tau, 1],
    [0, -b * d / (M * L), -b * (m + M) * g / (M * L), 1 / tau]
])
# print(np.linalg.eig(A))
# np.array([1.0, 0.99953272, 0.99513775, 1.00492952])
# np.array([
#     [1.00000000e+00, -9.73767702e-01, 1.31603805e-01, -1.09515682e-01],
#     [0.00000000e+00, 2.27510483e-01, -3.19945037e-01, -2.69930132e-01],
#     [0.00000000e+00, 3.82665589e-03, -3.56918366e-01, 3.59649831e-01],
#     [0.00000000e+00, -8.94057513e-04, 8.67712448e-01, 8.86451374e-01]
# ])
B = tau * np.array([
    [0],
    [1/M],
    [0],
    [b/(M*L)]
])
Q = np.eye(4) # state cost, 4x4 identity matrix
R = 0.0001 # control cost

def f(x, u):
    return A @ x + B @ u

#%% Traditional LQR
# lq = dlqr(A, B, Q, R)
# C = lq[0]

#%% Solve riccati equation
soln = dare(A * np.sqrt(gamma), B * np.sqrt(gamma), Q, R)
P = soln[0]
C = np.linalg.inv(R + gamma * B.T @ P @ B) @ (gamma * B.T @ P @ A)

#%% Construct snapshots of u from random agent and initial states x0
x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

perturbation = np.array([
    [0],
    [0],
    [np.random.normal(0, 0.05)],
    [0]
])
x = x0 + perturbation

N = 10000
action_range = 50
state_range = 10
U = np.random.rand(1,N) * action_range * np.random.choice(np.array([-1,1]), size=(1,N))
X0 = np.random.rand(4,N) * state_range * np.random.choice(np.array([-1,1]), size=(4,N))

#%% Construct snapshots of states following dynamics f
Y = f(X0, U)

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X0,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(2),
    regressor='ols'
)

#%% Define cost function

#%% Define cost
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    return mat

#%% Discretize all controls
u_bounds = np.array([[-action_range, action_range]])
step_size = 0.1
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1], step=step_size).reshape(1,-1)
All_U = np.round(All_U, decimals=1)
# All_U = U.reshape(1,-1) # continuous case is just original domain

#%% Learn control
gamma = 0.9
lamb = 0.1
lr = 1e-1
epsilon = 1e-2

algos = algorithmsv2.algos(
    X0,
    All_U,
    u_bounds[0],
    tensor,
    cost,
    gamma=gamma,
    epsilon=epsilon,
    bellman_error_type=0,
    learning_rate=lr,
    weight_regularization_bool=True,
    weight_regularization_lambda=lamb,
    optimizer='adam'
)
algos.w = np.load('bellman-weights.npy')
print("Weights before updating:", algos.w)
# bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
# print("Weights after updating:", algos.w)

# plt.plot(bellmanErrors)
# plt.show()
# plt.plot(gradientNorms)
# plt.show()

#%% Reset seed and compute initial x0s
np.random.seed(123)

num_episodes = 100
num_steps_per_episode = 1000
initial_Xs = np.random.rand(4,num_episodes) * state_range * np.random.choice(np.array([-1,1]), size=(4,num_episodes)) # random initial states

#%% Construct policy
All_U_range = np.arange(All_U.shape[1])

sigma_t = np.linalg.inv(R + B.T @ P @ B)
def policy(x, policyType):
    if policyType == 'learned':
        pis = algos.pis(x)[:,0]
        # Select action column at index sampled from policy distribution
        u_ind = np.random.choice(All_U_range, p=pis)
        u = np.vstack(
            All_U[:,u_ind]
        )
        return [u, u_ind]

    elif policyType == 'optimal':
        return [-C @ x, 0]

    elif policyType == 'optimalEntropy':
        return [np.random.normal(-C @ x, sigma_t), 0]

    elif policyType == 'random':
        return [np.random.rand(1,1)*(action_range)*np.random.choice(np.array([-1,1])),0] # sample random action
        #! Issue with log of policy density on action_range (-10, 10)

def policyDensity(u, u_ind, x, policyType):
    if policyType == 'learned':
        pi_term = algos.pis(x)[u_ind,0]
        return pi_term

    elif policyType == 'optimalEntropy':
        mu = -C @ x
        pi_term = np.exp((-(u-mu)**2)/(2*sigma_t**2))/(sigma_t*np.sqrt(2*np.pi))
        return pi_term

    elif policyType == 'random':
        pi_term = 1/(2*action_range)
        return pi_term

#%% Test policy by simulating system
# policy_type = 'learned'
# costs = np.empty((num_episodes))
# lamb = 1e-2 # 1.0?
opt_x0s = []
opt_x1s = []
opt_x2s = []
opt_x3s = []

learned_x0s = []
learned_x1s = []
learned_x2s = []
learned_x3s = []
for episode in range(1): #num_episodes
    x = np.vstack(initial_Xs[:,episode])
    opt_x0s.append(x[0,0])
    opt_x1s.append(x[1,0])
    opt_x3s.append(x[2,0])
    opt_x2s.append(x[3,0])
    learned_x0s.append(x[0,0])
    learned_x1s.append(x[1,0])
    learned_x2s.append(x[2,0])
    learned_x3s.append(x[3,0])
    # print("Initial x:", x)
    # cost_sum = 0
    for step in range(num_steps_per_episode):
        opt_u, opt_u_ind = policy(x, 'optimalEntropy')
        opt_x_prime = f(x, opt_u)

        opt_x0s.append(opt_x_prime[0,0])
        opt_x1s.append(opt_x_prime[1,0])
        opt_x2s.append(opt_x_prime[2,0])
        opt_x3s.append(opt_x_prime[3,0])

        learned_u, learned_u_ind = policy(x, 'learned')
        learned_x_prime = f(x, learned_u)

        learned_x0s.append(learned_x_prime[0,0])
        learned_x1s.append(learned_x_prime[1,0])
        learned_x2s.append(learned_x_prime[2,0])
        learned_x3s.append(learned_x_prime[3,0])

        # pis = algos.pis(x)[:,0]
        # (beta**step)*
        # cost_sum += (gamma**step)*(cost(x, u) + lamb*np.log(policyDensity(u, u_ind, x, policy_type)))

        # x = x_prime
        # if step%250 == 0:
        #     print("Current x:", x)
    # costs[episode] = cost_sum
# print("Mean cost per episode:", np.mean(costs)) # Cost should be minimized
plt.plot(np.arange(num_steps_per_episode+1), opt_x0s)
plt.plot(np.arange(num_steps_per_episode+1), opt_x1s)
plt.plot(np.arange(num_steps_per_episode+1), opt_x2s)
plt.plot(np.arange(num_steps_per_episode+1), opt_x3s)
plt.show()
plt.plot(np.arange(num_steps_per_episode+1), learned_x0s)
plt.plot(np.arange(num_steps_per_episode+1), learned_x1s)
plt.plot(np.arange(num_steps_per_episode+1), learned_x2s)
plt.plot(np.arange(num_steps_per_episode+1), learned_x3s)
plt.show()

#%%