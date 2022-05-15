#%% Imports
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

from control.matlab import lqr
from scipy.integrate import solve_ivp, odeint

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% True System Dynamics
state_dim = 4
state_column_shape = [state_dim,1]
action_dim = 1
action_column_shape = [action_dim,1]

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

action_range = np.arange(-20, 20, 0.1)
t_span = np.arange(0, 0.002, 0.0001)

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

# def pendcart_v2(x,tau,m,M,L,g,d,u):
#     x = x[:,0]
#     Sx = np.sin(x[2])
#     Cx = np.cos(x[2])
#     D = m*L*L*(M+m*(1-Cx**2))
#     dx = np.zeros(4)
#     dx[0] = x[1]*tau
#     dx[1] = tau*((1/D)*(-(m**2)*(L**2)*g*Cx*Sx + m*(L**2)*(m*L*(x[3]**2)*Sx - d*x[1])) + m*L*L*(1/D)*u)
#     dx[2] = x[3]*tau
#     dx[3] = tau*((1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*(x[3]**2)*Sx - d*x[1])) - m*L*Cx*(1/D)*u)
#     return (x + dx)

def continuous_f(action=None, policy=None):
    """
        INPUTS:
        action - action vector
    """

    def f_u(t, x):
        """
            INPUTS:
            t - timestep
            input - state vector
        """
        
        u = np.random.choice(action_range, size=action_column_shape)
        if action is not None:
            u = action
        if policy is not None:
            u = policy(x)

        x_prime = A @ np.vstack(x) + B @ u

        return np.array([
            x_prime[0,0],
            x_prime[1,0],
            x_prime[2,0],
            x_prime[3,0]
        ])

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action

    soln = solve_ivp(fun=continuous_f(action=u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

#%% Traditional LQR
K = lqr(A, B, Q, R)[0]

#%%
x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])
# w_r = np.zeros([4,1])

def random_policy(x):
    return np.random.choice(action_range, size=action_column_shape)

def optimal_policy(x):
    return -K @ (x - w_r[:, 0])

#%% Plot?
perturbation = np.array([
    [0],
    [0],
    [np.random.normal(0, 0.05)],
    [0]
])
x = x0 + perturbation
tspan = np.arange(0, 10, 0.0001)
_x = odeint(continuous_f(policy=optimal_policy), x[:, 0], tspan, tfirst=True)
# _x = odeint(pendcart, x[:, 0], tspan, args=(m, M, L, g, d, optimal_policy))

for i in range(state_dim):
    plt.plot(_x[:, i])
plt.show()

sys.exit(0)

#%% Construct Datasets
num_episodes = 200
num_steps_per_episode = 1000

seconds_per_step = 0.002

X = np.zeros([state_dim, num_episodes*num_steps_per_episode])
Y = np.zeros([state_dim, num_episodes*num_steps_per_episode])
U = np.zeros([action_dim, num_episodes*num_steps_per_episode])

for episode in range(num_episodes):
    perturbation = np.array([
            [0],
            [0],
            [np.random.normal(0, 0.05)],
            [0]
    ])
    x = x0 + perturbation

    for step in range(num_steps_per_episode):
        X[:, (episode*num_steps_per_episode)+step] = x[:, 0]
        u = random_policy(x)
        U[:, (episode*num_steps_per_episode)+step] = u
        # y = pendcart_v2(x,seconds_per_step,m,M,L,g,d,u)
        y = f(x, u)[:, 0]
        Y[:, (episode*num_steps_per_episode)+step] = y
        x = np.vstack(y)

#%% Koopman Tensor
state_order = 2
action_order = 2
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Training error
training_norms = np.zeros([num_episodes,num_steps_per_episode])
training_state_norms = np.zeros([num_episodes,num_steps_per_episode])
for episode in range(num_episodes):
    x = X[:, (episode*num_steps_per_episode)]
    for step in range(num_steps_per_episode):
        training_state_norms[episode,step] = utilities.l2_norm(x, np.zeros_like(x))
        u = U[:, (episode*num_steps_per_episode)+step]
        true_x_prime = Y[:, (episode*num_steps_per_episode)+step]
        predicted_x_prime = tensor.f(np.vstack(x), u)[:, 0]
        training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
        x = true_x_prime
average_training_l2_norm_per_episode = np.mean(np.sum(training_norms, axis=1))
training_state_mean_norm = np.mean(training_state_norms)
print(f'Average training l2 norm per episode over {num_episodes} episodes:', average_training_l2_norm_per_episode)
print(f'Normalized average training l2 norm per episode over {num_episodes} episodes:', average_training_l2_norm_per_episode/training_state_mean_norm)

#%% Testing error
testing_norms = np.zeros([num_episodes,num_steps_per_episode])
testing_state_norms = np.zeros([num_episodes,num_steps_per_episode])
for episode in range(num_episodes):
    perturbation = np.array([
            [0],
            [0],
            [np.random.normal(0, 0.05)],
            [0]
    ])
    x = x0 + perturbation

    for step in range(num_steps_per_episode):
        testing_state_norms[episode,step] = utilities.l2_norm(x, np.zeros_like(x))
        u = random_policy(x)
        # true_x_prime = pendcart_v2(x,seconds_per_step,m,M,L,g,d,u)
        true_x_prime = f(x, u)[:, 0]
        predicted_x_prime = tensor.f(x, u)[:, 0]
        testing_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
        x = np.vstack(true_x_prime)
average_testing_l2_norm_per_episode = np.mean(np.sum(testing_norms, axis=1))
testing_state_mean_norm = np.mean(testing_state_norms)
print(f'Average testing l2 norm per episode over {num_episodes} episodes:', average_testing_l2_norm_per_episode)
print(f'Normalized average testing l2 norm per episode over {num_episodes} episodes:', average_testing_l2_norm_per_episode/testing_state_mean_norm)

#%%