import gym
import numpy as np
import scipy as sp
# import torch

import sys
sys.path.append('../../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../../')
import cartpole_reward
import observables

#%% Initialize environment
env = gym.make('CartPole-v0')
# env = gym.make('env:CartPoleControlEnv-v0')

# def reward(xs, us):
#     return cartpole_reward.defaultCartpoleRewardMatrix(xs, us)
# def cost(xs, us):
#     return -reward(xs, us)

gamma = 0.99
eta = 0.001

w_r = np.zeros([4,1])

Q_ = np.array([
    [10, 0,  0, 0],
    [ 0, 1,  0, 0],
    [ 0, 0, 10, 0],
    [ 0, 0,  0, 1]
])
R = 0.1
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat

#%% Construct snapshots of u from random agent and initial states x0
N = 20000 # Number of datapoints
U = np.zeros([1,N])
X = np.zeros([4,N+1])
Y = np.zeros([4,N])
i = 0
while i < N:
    X[:,i] = env.reset()
    done = False
    while i < N and not done:
        U[0,i] = env.action_space.sample()
        Y[:,i], _, done, __ = env.step( int(U[0,i]) ) # np.array([int(U[0,i])])
        if not done:
            X[:,i+1] = Y[:,i]
        i += 1
X = X[:,:-1]

#%% Estimate Koopman tensor
order = 2
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(order),
    psi=observables.monomials(2),
    regressor='ols'
)

#%% Setup vars
state_dim = 4 # env.observation_space.shape[0]
M_plus_N_minus_ones = np.arange( (state_dim-1), order + (state_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * (state_dim-1) ) ) )
u_range = np.array([-5, 5])
step_size = 1 # 0.1
all_us = np.arange(u_range[0], u_range[1], step_size)

#%% Handy functions
def w_hat_t(xs, pi_t):
    phi_xs = tensor.phi(xs) # (dim_phi, batch_size)
    
    phi_xs_primes = np.zeros([phi_xs.shape[1], all_us.shape[0], phi_xs.shape[0]]) # (batch_size, num_actions, dim_phi)
    costs = np.zeros([phi_xs.shape[1], all_us.shape[0]])
    for j in range(xs.shape[1]):
        x = np.vstack(xs[:,j])
        for i in range(all_us.shape[0]):
            pi_t_response = pi_t( x, np.array([[all_us[i]]]) )
            phi_xs_primes[j,i] = ( tensor.phi_f(x, all_us[i]) * pi_t_response )[:, 0]
            costs[j,i] = -cost(x, np.array([[all_us[i]]]))[0,0] * pi_t_response
        
    expectation_term_1 = np.sum(phi_xs_primes, axis=0) # TODO: Check 'axis=' spec
    expectation_term_2 = np.sum(costs, axis=0) # TODO: Check 'axis=' spec

    return OLS(
        (phi_xs - gamma*expectation_term_1.data.numpy().T).T,
        np.array([[expectation_term_2.T.data.numpy()]])
    )

# def Q(x, u):
#     return -cost(x, u) + gamma*w_hat_t(x).T @ tensor.phi_f(x, u)

def inner_pi_us(us, xs, w_hat):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs)
    inner_pi_us = -(cost(xs, us).T + gamma*(w_hat.T @ phi_x_primes)[:,0]) # TODO: negative cost?
    return inner_pi_us

def pi_ts(xs, w_hat, pi_tminus1s):
    # delta = 1e-25
    inner_pi_us_response = inner_pi_us(np.array([all_us]), xs, w_hat)
    print("inner_pi_us_response shape:", inner_pi_us_response.shape)
    inner_pi_us_response = np.real(inner_pi_us_response)
    max_inner_pi_u = np.amax(inner_pi_us_response, axis=0)
    print("max_inner_pi_u shape:", max_inner_pi_u.shape)
    diff = inner_pi_us_response - max_inner_pi_u
    print("diff shape:", diff.shape)
    pi_us = np.exp(diff) * pi_tminus1s(xs) #+ delta
    print("pi_us shape:", pi_us.shape)
    Z_x = np.sum(pi_us, axis=0)
    print("Z_x shape:", Z_x)

    print("pi_us / Z_x shape:", (pi_us / Z_x).shape)

    return pi_us / Z_x # TODO: normalization?

#%% ###Policy Iteration###

def pi_0(u, x):
    return 1 / all_us.shape[0]

pi_0s = np.zeros([X.shape[1],all_us.shape[0]])
for i in range(X.shape[1]):
    x = np.vstack(X[:,i])
    for j in range(all_us.shape[0]):
        pi_0s[i,j] = pi_0(all_us[j], x)

batch_size = 512
iter = 10000
pi = pi_0
for i in range(iter):
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # X.shape[0] x batch_size
    w_hat = w_hat_t(x_batch, pi)
    pi = pi_ts(x_batch, w_hat, pi) # TODO: This isn't quite right I think