'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 8: Implementing Policy Gradients and Policy Optimization
Author: Yuxi (Hayden) Liu
'''

import gym
import numpy as np
import torch
import torch.nn as nn

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

import sys
sys.path.append('../../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../../')
import cartpole_reward
import observables

#%% Environment settings
# env_string = 'CartPole-v0'
env_string = 'env:CartPoleControlEnv-v0'
env = gym.make(env_string)

step_size = 1.0 if env_string == 'CartPole-v0' else 2
all_us = torch.arange(0, 1+step_size, step_size) if env_string == 'CartPole-v0' else  torch.arange(-10, 10+step_size, step_size)

#%% Reward function
def reward(xs, us):
    return cartpole_reward.defaultCartpoleRewardMatrix(xs, np.array([us])).T
def cost(xs, us):
    return -reward(xs, us)

# w_r = np.zeros([4,1])
Q_ = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
# Q_diag = np.diag(Q_)
# Q_ = np.array([10.0, 1.0, 10.0, 1.0])
R = 0.1
# def lqr_cost(x, u):
#     return x.T @ Q_ @ x + (u.T * R) @ u
# def cost(x, u):
#     # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
#     # _x = x - w_r
#     mat = np.vstack(np.diag(x.T @ Q_ @ x)) + np.power(u, 2)*R
#     return mat # (xs.shape[1], us.shape[1])
# def reward(x, u):
#     return -cost(x, u)

#%% Policy function as PyTorch model
class Policy:
    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        (torch.distributions.Distribution)

        s_t (np.ndarray): the current state
        '''
        raise NotImplementedError

    def act(self, s_t):
        '''
        s_t (np.ndarray): the current state
        Because of environment vectorization, this will produce
        E actions where E is the number of parallel environments.
        '''
        a_t = self.pi(s_t).sample()
        return a_t

    def learn(self, states, actions, returns):
        '''
        states (np.ndarray): the list of states encountered during
                             rollout
        actions (np.ndarray): the list of actions encountered during
                              rollout
        returns (np.ndarray): the list of returns encountered during
                              rollout

        Because of environment vectorization, each of these has first
        two dimensions TxE where T is the number of time steps in the
        rollout and E is the number of parallel environments.
        '''
        actions = torch.tensor(actions)
        returns = torch.tensor(returns)

        log_prob = self.pi(states).log_prob(actions)
        loss = torch.mean(-log_prob*returns)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class DiagonalGaussianPolicy(Policy):
    def __init__(self, env, lr=1e-2):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        '''
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.shape[0]

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self.N, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.M)
        ).double()

        self.log_sigma = torch.ones(self.M, dtype=torch.double, requires_grad=True)

        self.opt = torch.optim.Adam(list(self.mu.parameters()) + [self.log_sigma], lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        s_t = torch.as_tensor(s_t).double()
        mu = self.mu(s_t)
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        return pi

def reinforce(env, agent, n_episode, gamma=1.0):
    """
        REINFORCE algorithm
        @param env: Gym environment
        @param estimator: policy network
        @param n_episode: number of episodes
        @param gamma: the discount factor
    """
    w_hat = np.zeros([15,1])
    for episode in range(n_episode):
        states = []
        actions = []
        rewards = []
        state = env.reset()

        while len(rewards) < 300:
            action = agent.act(state)
            next_state, _, is_done, __ = env.step(action.data.numpy())

            states.append(state)
            actions.append(action.data.numpy())

            curr_reward = reward(np.vstack(state), np.array([[action.data.numpy()]]))[0,0]
            total_reward_episode[episode] += curr_reward
            rewards.append(curr_reward)

            if is_done or len(rewards) == 300:
                returns = torch.zeros([len(rewards)])
                # Gt = 0
                for i in range(len(rewards)-1, -1, -1):
                    # Gt = rewards[i] + (gamma * Gt)
                    # returns[i] = Gt
                    # print("Gt:", Gt)
                    Q_val = Q(
                        np.vstack(states[i]),
                        np.array([actions[i]]),
                        w_hat
                    )
                    returns[i] = Q_val
                    # print("Q_val:", Q_val)

                # if episode == 0 or (episode+1) % 100 == 0:
                #     print(returns)

                returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

                agent.learn(np.array(states), np.array(actions), np.array(returns))
                if episode == 0 or (episode+1) % 100 == 0:
                    # torch.save(agent, 'gaussian-policy-iteration.pt')
                    print(f"Episode: {episode+1}, total reward: {total_reward_episode[episode]}, num steps takes: {len(rewards)}")

                break

            state = next_state

        w_hat = w_hat_t()

#%%
n_state = env.observation_space.shape[0]
lr = 0.003
# agent = DiagonalGaussianPolicy(env, lr=1e-2)
agent = DiagonalGaussianPolicy(env, lr=lr)
# agent = torch.load('gaussian-policy-iteration.pt')

n_episode = 3000 # Completely converged at 2000 episodes for original code
gamma = 0.99
total_reward_episode = [0] * n_episode

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

#%% Learn Koopman Tensor
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

#%% Estimate Q function for current policy
w_hat_batch_size = 2**14 # 2**9
sample_u_batch_size = 2**10
standard_normal_distribution = torch.distributions.Normal(0,1)
sampled_actions = standard_normal_distribution.sample([sample_u_batch_size])
sampled_action_probabilities = torch.exp(standard_normal_distribution.log_prob(sampled_actions))
# print( (torch.ones([w_hat_batch_size,sample_u_batch_size]) * sampled_actions).shape ) # (w_hat_batch_size, sample_u_batch_size)
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        distributions = agent.pi(x_batch.T)
        # actions = distributions.sample()
        pi_response = torch.zeros([sample_u_batch_size, w_hat_batch_size])
        for i in range(sample_u_batch_size):
            pi_response[i] = torch.exp(distributions.log_prob(torch.tensor([sampled_actions[i]]))) / sampled_action_probabilities[i]
        pi_response = pi_response.data.numpy()

    phi_x_prime_batch = tensor.K_(np.array([sampled_actions.data.numpy()])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response) # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = np.einsum('wu,uw->wu', reward(x_batch, np.array([sampled_actions.data.numpy()])), pi_response) # (w_hat_batch_size, all_us.shape[0])
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    )

    return w_hat

def Q(x, u, w_hat_t):
    return (reward(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u))[0,0]

#%%
reinforce(env, agent, n_episode, gamma)

# import matplotlib.pyplot as plt
# plt.plot(total_reward_episode)
# plt.title('Episode reward over time')
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()

#%% Test policy in environment
num_episodes = 10#00
def watch_agent():
    rewards = torch.zeros([num_episodes])
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            env.render()
            with torch.no_grad():
                action = agent.act(state)
            state, _, done, __ = env.step(action.data.numpy())
            step += 1
            if done or step >= 200:
                rewards[episode] = step
                # print("Reward:", step)
    env.close()
    print(f"Mean reward per episode over {num_episodes} episodes:", torch.mean(rewards))
watch_agent()

#%%