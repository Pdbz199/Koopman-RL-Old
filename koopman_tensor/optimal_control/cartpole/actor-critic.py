import gym, os
from itertools import count
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from scipy.special import comb

import sys
sys.path.append('../../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../../')
import cartpole_reward
import observables

#%% Dynamics variables
# env_string = 'CartPole-v0'
env_string = 'env:CartPoleControlEnv-v0'
env = gym.make(env_string).unwrapped

state_order = 2
action_order = 2

# state_dim = env.observation_space.shape[0]
state_dim = 4
action_dim = 1
state_M_plus_N_minus_ones = np.arange( (state_dim-1), state_order + (state_dim-1) + 1 )
phi_dim = int( np.sum( comb( state_M_plus_N_minus_ones, np.ones_like(state_M_plus_N_minus_ones) * (state_dim-1) ) ) )
action_M_plus_N_minus_ones = np.arange( (action_dim-1), action_order + (action_dim-1) + 1 )
psi_dim = int( np.sum( comb( action_M_plus_N_minus_ones, np.ones_like(action_M_plus_N_minus_ones) * (action_dim-1) ) ) )
step_size = 1 if env_string == 'CartPole-v0' else 3.0 #0.1
u_range = np.array([0, 1+step_size]) if env_string == 'CartPole-v0' else np.array([-15, 15+step_size])
all_us = np.arange(u_range[0], u_range[1], step_size)
all_us = np.round(all_us, decimals=2)
gamma = 0.99
lr = 0.0001

#%% Cost/reward functions
# def reward(xs, us):
#     return cartpole_reward.defaultCartpoleRewardMatrix(xs, us)
# def cost(xs, us):
#     return -reward(xs, us)

w_r = np.zeros([4,1])
Q_ = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat.T
def reward(x, u):
    return -cost(x, u)

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
        action = int(U[0,i])
        Y[:,i], _, done, __ = env.step(action)
        if not done:
            X[:,i+1] = Y[:,i]
        i += 1
X = X[:,:-1]

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Model definition
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

policy_model = None
if os.path.exists('actor-critic-policy.pkl'):
    policy_model = torch.load('actor-critic-policy.pkl')
else:
    policy_model = torch.nn.Sequential(
        torch.nn.Linear(state_dim, all_us.shape[0]),
        torch.nn.Softmax(dim=-1)
    )
    # Initialize weights with 0s
    policy_model.apply(init_weights)

#%% Q function
w_hat_batch_size = 2**14
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy_model(torch.from_numpy(x_batch).float().T).T # (all_us.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_us])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = phi_x_prime_batch * \
                                pi_response.reshape(
                                    pi_response.shape[0],
                                    1,
                                    pi_response.shape[1]
                                ).data.numpy() # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = reward(x_batch, np.array([all_us])) * pi_response.data.numpy() # (all_us.shape[0], w_hat_batch_size)
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=0) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    ) # (phi_dim, 1)

    return w_hat

def Q(x, u, w_hat_t):
    return reward(x, u).T + gamma*w_hat_t.T @ tensor.phi_f(x, u)[:,:,0].T

# def compute_returns(next_value, rewards, masks, gamma=0.99):
def compute_returns(next_value, rewards, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R #* masks[step]
        returns.insert(0, R)
    return returns

def critic(x, w_hat):
    _x = np.vstack(x)
    _all_us = np.array([all_us])
    v = torch.sum(
        torch.Tensor( Q( _x, _all_us, w_hat ) ) * policy_model( torch.Tensor(x) ).detach().reshape(1,all_us.shape[0]),
        axis=1
    )
    return v

def trainIters(n_iters):
    optimizerA = optim.Adam(policy_model.parameters())
    w_hat = np.zeros([phi_dim,1])
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        # masks = []

        for i in count():
            # env.render()
            dist = policy_model(torch.Tensor(state))
            value = critic(state, w_hat)
            # print("Value:", value)

            action_index = torch.multinomial(dist, 1).item()
            action = all_us[action_index].item()
            log_prob = torch.log(dist[action_index]).reshape(1,1)

            next_state, curr_reward, done, __ = env.step(action)
            # curr_reward = reward(np.vstack(state), action)[0,0]

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(curr_reward)
            # rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            # masks.append(torch.tensor([1-done], dtype=torch.float))

            state = next_state

            if done:
                if iter == 0 or (iter+1) % 100 == 0:
                    print('Iteration: {}, Score: {}'.format(iter+1, i))
                break

        next_value = critic(next_state, w_hat)
        # print("Next value:", next_value)
        # returns = compute_returns(next_value, rewards, masks)
        returns = compute_returns(next_value, rewards)

        log_probs = torch.cat(log_probs)
        returns = torch.Tensor(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()

        # Update actor
        optimizerA.zero_grad()
        actor_loss.backward()
        optimizerA.step()

        # Update critic
        w_hat = w_hat_t()

    torch.save(policy_model, 'actor-critic-policy.pkl')
    # env.close()

if __name__ == '__main__':
    trainIters(n_iters=5000)