#%% Imports
import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("env:CartPoleControlEnv-v0")
np.random.seed(123)

import sys
sys.path.append('../../')
from tensor import KoopmanTensor
sys.path.append('../../../')
# import algorithmsv2
import algorithmsv2_parallel as algorithmsv2
import cartpole_reward
import observables

from control import dare, dlqr
from scipy.integrate import quad_vec, odeint
from scipy.stats import norm

#%% System dynamics
gamma = 0.99
lamb = 0.000001
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
A = np.array([
    [1.0, 0.02,  0.0,        0.0 ],
    [0.0, 1.0,  -0.01434146, 0.0 ],
    [0.0, 0.0,   1.0,        0.02],
    [0.0, 0.0,   0.3155122,  1.0 ]
])
B = np.array([
    [0],
    [0.0195122],
    [0],
    [-0.02926829]
])

Q = np.array([
    [10,   0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1

# Cartpole A, B, Q, and R matrices from Steve's databook v2
# mass_pole = 1.0
# mass_cart = 5.0
# pole_position = 1.0
# pole_length = 2.0
# gravity = -10.0
# cart_damping = 1.0
# continuous_A = np.array([
#     [0.0, 1.0, 0.0, 0.0],
#     [0.0, -cart_damping / mass_cart, pole_position * mass_pole * gravity / mass_cart, 0.0],
#     [0.0, 0.0, 0.0, 1.0],
#     [0.0, -pole_position * cart_damping / mass_cart * pole_length, -pole_position * (mass_pole + mass_cart) * gravity / mass_cart * pole_length, 0.0]
# ])
# delta_t = 0.02
# A = np.exp(continuous_A * delta_t)
# continuous_B = np.array([
#     [0.0],
#     [1.0 / mass_cart],
#     [0.0],
#     [pole_position / mass_cart * pole_length]
# ])
# B = quad_vec(lambda tau: np.exp(continuous_A * tau) @ continuous_B, 0, delta_t)[0]
# t_span = np.arange(0, 0.02, 0.0001)
# B = odeint(lambda tau: np.exp(continuous_A * tau) @ continuous_B, continuous_B, t_span)[0][-1]
# Q = np.eye(4)
# R = 0.0001
# Reference state
# w_r = np.array([
#     [1],
#     [0],
#     [np.pi],
#     [0]
# ])
# w_r = np.zeros([4,1])
# def f(x, u):
#     return A @ x + B @ u

#%% Solve riccati equation
# C = dlqr(A, B, Q, R)[0]
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
# sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)

num_episodes = 1000
steps_per_episode = np.zeros([num_episodes])
costs_per_episode = np.zeros([num_episodes])
for episode in range(num_episodes):
    x = np.vstack(env.reset())
    done = False
    step = 0
    costs = []
    while not done and step < 200:
        # env.render()
        u = -C @ x
        print(u)
        x_prime,cost,done,__ = env.step(u[0,0])
        costs.append(cost)
        x = np.vstack(x_prime)
        step += 1
    steps_per_episode[episode] = step
    costs_per_episode[episode] = np.sum(costs)
# env.close()
print(f"Average steps per episode over {num_episodes} episodes:", np.mean(steps_per_episode))
print(f"Average cost per episode over {num_episodes} episodes:", np.mean(costs_per_episode))

#%%