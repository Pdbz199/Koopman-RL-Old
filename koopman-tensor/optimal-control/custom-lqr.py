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

from control import dare #, dlqr
from scipy.integrate import quad_vec, odeint
from scipy.stats import norm

#%% System dynamics
gamma = 0.99
lamb = 0.000001 # 0.1
# gamma = 0.5
# lamb = 1.0
# gamma = 0.5
# lamb = 0.6
# gamma = 0.9
# lamb = 1.0

# Check if Cartpole A matrix is within eigenvalue range
# A = np.array([
#     [1, 0.02,  0,          0],
#     [0, 1,    -0.01434146, 0],
#     [0, 0,     1,          0.02],
#     [0, 0,     0.3155122,  1]
# ])
# W,V = np.linalg.eig(A)
# print("Cartpole eigenvalues of A:", W)
# print("Cartpole eigenvectors of A:", V)

# A = np.zeros([2,2])
# max_real_eigen_val = 1.0
# while max_real_eigen_val >= 1.0 or max_real_eigen_val <= 0.7:
#     Z = np.random.rand(2,2)
#     A = Z.T @ Z
#     W,V = np.linalg.eig(A)
#     max_real_eigen_val = np.max(np.real(W))
# print("A:", A)
# A = np.array([
#     [0.5, 0.0],
#     [0.0, 0.3]
# ], dtype=np.float64)
# A = np.array([
#     [2.0, 0.0],
#     [0.0, 1.2]
# ], dtype=np.float64)
# B = np.array([
#     [1.0], # Try changing so that we get dampening on control
#     [1.0]
# ], dtype=np.float64)
# Q = np.array([
#     [1.0, 0.0],
#     [0.0, 1.0]
# ], dtype=np.float64) #* gamma
# R = 1
# R = np.array([[1.0]], dtype=np.float64) #* gamma

# Cartpole A, B, Q, and R matrices from Wen's homework
# A = np.array([
#     [1.0, 0.02,  0.0,        0.0 ],
#     [0.0, 1.0,  -0.01434146, 0.0 ],
#     [0.0, 0.0,   1.0,        0.02],
#     [0.0, 0.0,   0.3155122,  1.0 ]
# ])
# B = np.array([
#     [0],
#     [0.0195122],
#     [0],
#     [-0.02926829]
# ])

# Q = np.array([
#     [10, 0,  0, 0],
#     [ 0, 1,  0, 0],
#     [ 0, 0, 10, 0],
#     [ 0, 0,  0, 1]
# ])
# Q = np.array([
#     [10, 0],
#     [ 0, 1]
# ])
# R = 0.1

# Cartpole A, B, Q, and R matrices from Steve's databook v2
mass_pole = 1.0
mass_cart = 5.0
pole_position = 1.0
pole_length = 2.0
gravity = -10.0
cart_damping = 1.0
continuous_A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -cart_damping / mass_cart, pole_position * mass_pole * gravity / mass_cart, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, -pole_position * cart_damping / mass_cart * pole_length, -pole_position * (mass_pole + mass_cart) * gravity / mass_cart * pole_length, 0.0]
])
delta_t = 0.02
A = np.exp(continuous_A * delta_t)
continuous_B = np.array([
    [0.0],
    [1.0 / mass_cart],
    [0.0],
    [pole_position / mass_cart * pole_length]
])
B = quad_vec(lambda tau: np.exp(continuous_A * tau) @ continuous_B, 0, delta_t)[0]
# t_span = np.arange(0, 0.02, 0.0001)
# B = odeint(lambda tau: np.exp(continuous_A * tau) @ continuous_B, continuous_B, t_span)[0][-1]
print(B)
Q = np.eye(4)
R = 0.0001
# Reference state
w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])
# w_r = np.zeros([4,1])

def f(x, u):
    return A @ x + B @ u



#%% Traditional LQR
# lq = dlqr(A, B, Q, R)
# C = lq[0]
# lq[0] == [[ 8.96688317 -6.28428936]]
# lq[1] == [[ 79.40499383 -70.43811066]
#           [-70.43811066  64.1538213 ]]
# lq[2] == [-1.474176 +0.j -0.4084178+0.j]

#%% Solve riccati equation
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)

#! Condition number check, rank check
#! Controlability of the matrices
W,V = np.linalg.eig(A)
print("Eigenvalues of A:", W)
W,V = np.linalg.eig(A - B@C)
print("Eigenvalues of (A - BC):", W)

#%% Construct snapshots of u from random agent and initial states x0
N = 20000
action_range = 25
state_range = 25
U = np.random.rand(1,N)*action_range*np.random.choice(np.array([-1,1]), size=(1,N))
X0 = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))

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
# def cost(x, u):
#     return x.T @ Q @ x + u.T @ R @ u
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    x_r = x - w_r
    return np.vstack(np.diag(x_r.T @ Q @ x_r)) + np.power(u, 2)*R

#%% Discretize all controls
u_bounds = np.array([[-action_range, action_range]])
step_size = 0.1 # 0.01
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1], step=step_size).reshape(1,-1)
All_U = np.round(All_U, decimals=1)
# All_U = U.reshape(1,-1) # continuous case is just original domain

#%% Learn control
algos = algorithmsv2.algos(
    X0,
    All_U,
    u_bounds[0],
    tensor,
    cost,
    gamma=gamma,
    epsilon=1e-2,
    bellman_error_type=0,
    learning_rate=1e-1,
    weight_regularization_bool=True,
    weight_regularization_lambda=lamb,
    optimizer='adam',
    load=True
)
# algos.w = np.load('bellman-weights.npy')
# algos.w = np.array([
#     [-9.63500888e+00],
#     [ 6.44128461e-06],
#     [ 5.91321286e-06],
#     [ 1.10210390e+00],
#     [-4.33773261e-02],
#     [ 1.03527409e+00]
# ]) # epsilon = 0.01
# algos.w = np.array([
#     [-9.63863338e+00],
#     [-4.31300696e-05],
#     [-3.47747365e-06],
#     [ 1.10211780e+00],
#     [-4.33771878e-02],
#     [ 1.03530115e+00]
# ]) # epsilon = 0.001
print("Weights before updating:", algos.w)
# bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
# print("Weights after updating:", algos.w)

# def reward(x, u):
#     return -cost(x, u)
# theta = algos.REINFORCE(f, reward, sigma_t)
# print(theta)

# plt.plot(bellmanErrors)
# plt.show()
# plt.plot(gradientNorms)
# plt.show()

#%% Reset seed and compute initial x0s
np.random.seed(123)

num_episodes = 1#00
num_steps_per_episode = 100
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
np.random.rand(A.shape[0],num_episodes)
initial_Xs = np.random.rand(A.shape[0],num_episodes)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],num_episodes)) # random initial states

#%% Construct policy
All_U_range = np.arange(All_U.shape[1])

def policy(x, policyType):
    if policyType == 'learned':
        pis = algos.pis(x)[2][:,0]
        # Select action column at index sampled from policy distribution
        u_ind = np.random.choice(All_U_range, p=pis)
        u = np.vstack(
            All_U[:,u_ind]
        )
        return [u, u_ind]

    elif policyType == 'optimal':
        return [-C @ (x - w_r), 0]

    elif policyType == 'optimalEntropy':
        return [np.random.normal(-C @ (x - w_r), sigma_t), 0]

    elif policyType == 'random':
        return [np.random.rand(1,1)*(action_range)*np.random.choice(np.array([-1,1])),0] # sample random action
        #! Issue with log of policy density on action_range (-10, 10)

sigma_squared = np.power(sigma_t, 2)
def pi(x, theta):
    return np.random.normal(tensor.phi(x).T @ theta, sigma_squared)

def policyDensity(u, u_ind, x, policyType):
    if policyType == 'learned':
        pi_term = algos.pis(x)[2][u_ind,0]
        return pi_term

    elif policyType == 'optimalEntropy':
        mu = -C @ (x - w_r)
        pi_term = np.exp((-(u-mu)**2)/(2*sigma_t**2))/(sigma_t*np.sqrt(2*np.pi))
        return pi_term

    elif policyType == 'random':
        pi_term = 1/(2*action_range)
        return pi_term

#%% Test policy by simulating system
import gym
env = gym.make("env:CartPoleControlEnv-v0")
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

optimal_cost_per_episode = np.zeros([num_episodes])
learned_cost_per_episode = np.zeros([num_episodes])
for episode in range(num_episodes):
    initial_Xs[:,episode] = env.reset()
    opt_x = np.vstack(initial_Xs[:,episode])
    learned_x = np.vstack(initial_Xs[:,episode])
    opt_x0s.append(opt_x[0,0])
    learned_x0s.append(learned_x[0,0])
    opt_x1s.append(opt_x[1,0])
    learned_x1s.append(learned_x[1,0])
    opt_x2s.append(opt_x[2,0])
    learned_x2s.append(learned_x[2,0])
    opt_x3s.append(opt_x[3,0])
    learned_x3s.append(learned_x[3,0])
    
    for step in range(num_steps_per_episode):
        opt_u, opt_u_ind = policy(opt_x, 'optimalEntropy')
        opt_x_prime = f(opt_x, opt_u)

        opt_x0s.append(opt_x_prime[0,0])
        opt_x1s.append(opt_x_prime[1,0])
        opt_x2s.append(opt_x_prime[2,0])
        opt_x3s.append(opt_x_prime[3,0])

        optimal_cost_per_episode[episode] += cost(opt_x, opt_u) # (gamma**step)*(cost(opt_x, opt_u) + lamb*np.log(policyDensity(opt_u, opt_u_ind, opt_x, 'optimalEntropy')))

        learned_u, learned_u_ind = policy(learned_x, 'learned')
        # learned_u = pi(learned_x, theta)
        learned_x_prime = f(learned_x, learned_u)

        learned_x0s.append(learned_x_prime[0,0])
        learned_x1s.append(learned_x_prime[1,0])
        learned_x2s.append(learned_x_prime[2,0])
        learned_x3s.append(learned_x_prime[3,0])

        learned_cost_per_episode[episode] += cost(learned_x, learned_u) # (gamma**step)*(cost(learned_x, learned_u) + lamb*np.log(policyDensity(learned_u, learned_u_ind, learned_x, 'learned')))

        opt_x = opt_x_prime
        learned_x = learned_x_prime

print(f"Mean cost per optimal episode over {num_episodes} episode(s):", np.mean(optimal_cost_per_episode))
print(f"Mean cost per learned episode over {num_episodes} episode(s):", np.mean(learned_cost_per_episode))

#%% Plot action distribution
x = np.vstack(env.reset())
# x = np.vstack(initial_Xs[:,0])
fig, axs = plt.subplots(2)
fig.suptitle('Policy distribution')
axs[0].set_title('Optimal entropy distribution')
axs[0].plot(All_U[0], norm.pdf(All_U[0], (-C @ (x - w_r))[0,0], sigma_t[0,0]))
axs[1].set_title('Learned distribution')
axs[1].plot(All_U[0], algos.pis(x)[2][:,0])
plt.show()

#%% Plot states over time
if num_episodes == 1:
    x_axis = np.arange(num_steps_per_episode+1)
    fig, axs = plt.subplots(2)

    axs[0].set_title("States over time for optimal agent")
    axs[0].set(xlabel='Timestep', ylabel='State value')
    axs[0].plot(x_axis, opt_x0s, label='x_0')
    axs[0].plot(x_axis, opt_x1s, label='x_1')
    axs[0].plot(x_axis, opt_x2s, label='x_2')
    axs[0].plot(x_axis, opt_x3s, label='x_3')

    axs[1].set_title("States over time for learned agent")
    axs[1].set(xlabel='Timestep', ylabel='State value')
    axs[1].plot(x_axis, learned_x0s, label='x_0')
    axs[1].plot(x_axis, learned_x1s, label='x_1')
    axs[1].plot(x_axis, learned_x2s, label='x_2')
    axs[1].plot(x_axis, learned_x3s, label='x_3')

    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.show()

#%% Try in gym environment
# import gym
# env = gym.make("env:CartPoleControlEnv-v0")
for episode in range(num_episodes):
    x = np.vstack(env.reset())
    done = False
    step = 0
    while not done and step < 200:
        env.render()
        # u, u_ind = policy(x, 'learned')
        u, u_ind = policy(x, 'optimalEntropy')
        x_prime,_,done,__ = env.step(u[:,0])
        x = np.vstack(x_prime)
        step += 1
env.close()

#%%