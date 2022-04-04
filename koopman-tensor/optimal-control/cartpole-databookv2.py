#%% Imports
import gym
import matplotlib.pyplot as plt
import numpy as np

from control.matlab import lqr
from scipy import integrate

#%% Dynamics
m = 1 # pendulum mass
M = 5 # cart mass
L = 2 # pendulum length
g = -10 # gravitational acceleration
d = 1 # (delta) cart damping

b = 1 # pendulum up (b = 1)

A = np.array([
    [0,1,0,0],
    [0,-d/M,b*m*g/M,0],
    [0,0,0,1],
    [0,- b*d/(M*L),-b*(m+M)*g/(M*L),0]
])
B = np.array([
    [0],
    [1/M],
    [0],
    [b/(M*L)]
])
Q = np.eye(4) # state cost, 4x4 identity matrix
R = 0.0001 # control cost

# def f(x, u):
#     return A @ x + B @ u

def pendcart(x,t,m,M,L,g,d,uf):
    u = uf(x) # evaluate anonymous function at x
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx**2))
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx

#%% Traditional LQR
K = lqr(A, B, Q, R)[0]

#%%
x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

perturbation = np.array([
        [0],
        [0],
        [np.random.normal(0, 0.1)], # np.random.normal(0, 0.05)
        [0]
])
x = x0 + perturbation

w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])

action_range = np.arange(-500, 500)
def random_policy(x):
    return np.array([[np.random.choice(action_range)]])

def optimal_policy(x):
    return -K @ (x - w_r[:, 0])

#%%
tspan = np.arange(0, 10, 0.001)
_x = integrate.odeint(pendcart, x[:, 0], tspan, args=(m, M, L, g, d, random_policy))

#%%
for i in range(4):
    # plt.close()
    plt.plot(_x[:, i])
plt.show()

#%%