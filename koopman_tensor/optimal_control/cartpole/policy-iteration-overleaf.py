import gym
import numpy as np
import scipy as sp
# import torch

import sys
sys.path.append('../../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../../')
# import cartpole_reward
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
action_dim = 1
M_plus_N_minus_ones = np.arange( (state_dim-1), order + (state_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * (state_dim-1) ) ) )
u_range = np.array([-5, 5])
step_size = 1 # 0.1
all_us = np.arange(u_range[0], u_range[1]+step_size, step_size)

#%% Handy functions
w_hat_batch_size = X.shape[1] # was 2**10
def w_hat_t(pi_t):
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    pi_response  = pi_t( x_batch, np.array([all_us]) )

    phi_x_prime_batch = tensor.K_(np.array([all_us])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = phi_x_prime_batch * \
                                pi_response.reshape(
                                    pi_response.shape[0],
                                    1,
                                    pi_response.shape[1]
                                ) # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    cost_batch_prob = cost(x_batch, np.array([all_us])) * pi_response # (all_us.shape[0], w_hat_batch_size)
    expectation_term_2 = np.array([
        np.sum(cost_batch_prob, axis=0) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    return OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    )

def Q(x, u):
    return cost(x, u) + gamma*w_hat_t(x).T @ tensor.phi_f(x, u)

def inner_pi_us(us, xs):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs)
    inner_pi_us_response = -(cost(xs, us).T + gamma*(w_hat.T @ phi_x_primes)[:,0])
    return inner_pi_us_response

def pi_t_plus_1(w_hat, pi_t):
    # delta = 1e-25
    def g(us, xs):
        inner_pi_us_response = inner_pi_us(us, xs, w_hat) #(np.array([all_us]), xs, w_hat)
        print("inner_pi_us_response shape:", inner_pi_us_response.shape)
        inner_pi_us_response = np.real(inner_pi_us_response)
        max_inner_pi_u = np.amax(inner_pi_us_response, axis=0)
        print("max_inner_pi_u shape:", max_inner_pi_u.shape)
        diff = inner_pi_us_response - max_inner_pi_u
        print("diff shape:", diff.shape)
        pi_us = np.exp(diff) * pi_t(xs) # + delta
        print("pi_us shape:", pi_us.shape)
        Z_x = np.sum(pi_us, axis=0)
        print("Z_x shape:", Z_x)

        print("pi_us / Z_x shape:", (pi_us / Z_x).shape)
        return pi_us / Z_x
    
    return g

#%% ###Policy Iteration###

def pi_0(us, xs):
    return np.ones([us.shape[1], xs.shape[1]]) * 1 / all_us.shape[0]

all_pis = [pi_0]

batch_size = 512
iter = 10000
for i in range(iter):
    pi_t = all_pis[-1]
    w_hat = w_hat_t(pi_t)
    all_pis.append(pi_t_plus_1(w_hat, pi_t))
all_pis = np.array(all_pis)

#%% Test evaluation
pi_T = all_pis[-1]
print( pi_T(U[:,100], X[:,100]) )