'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 8: Implementing Policy Gradients and Policy Optimization
Author: Yuxi (Hayden) Liu
'''

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import dare
# from scipy.integrate import quad_vec

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../')
import cartpole_reward
import observables

#%% Environment settings
# env_string = 'CartPole-v0'
# env_string = 'env:CartPoleControlEnv-v0'
# env = gym.make(env_string)

# A = np.zeros([2,2])
# max_real_eigen_val = 1.0
# # while max_real_eigen_val >= 1.0 or max_real_eigen_val <= 0.7:
# while max_real_eigen_val >= 2.0 or max_real_eigen_val <= 1.5:
#     Z = np.random.rand(2,2)
#     A = Z.T @ Z
#     W,V = np.linalg.eig(A)
#     max_real_eigen_val = np.max(np.real(W))
# print("A:", A)
# A = np.array([
#     [0.5, 0.0],
#     [0.0, 0.3]
# ], dtype=np.float64)
# B = np.array([
#     [1.0],
#     [1.0]
# ], dtype=np.float64)

# def f(x, u):
#     return A @ x + B @ u

mass_pole = 1.0
mass_cart = 5.0
pole_position = 1.0
pole_length = 2.0
gravity = -10.0
cart_damping = 1.0
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -cart_damping / mass_cart, pole_position * mass_pole * gravity / mass_cart, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, -pole_position * cart_damping / mass_cart * pole_length, -pole_position * (mass_pole + mass_cart) * gravity / mass_cart * pole_length, 0.0]
])
delta_t = 0.02
B = np.array([
    [0.0],
    [1.0 / mass_cart],
    [0.0],
    [pole_position / mass_cart * pole_length]
])

x0 = np.array([
    [-1],
    [0],
    [np.pi],
    [0]
])

def f(x, u):
    return x + (A @ x + B @ u)*delta_t

step_size = 0.01
all_us = torch.arange(-20, 20+step_size, step_size)

#%% Reward function
# def reward(xs, us):
#     return cartpole_reward.defaultCartpoleRewardMatrix(xs, np.array([us])).T
# def cost(xs, us):
#     return -reward(xs, us)

# w_r = np.zeros([A.shape[0],1])
# Q_ = np.array([
#     [1.0, 0.0],
#     [0.0, 1.0]
# ], dtype=np.float64)
# Q_diag = np.diag(Q_)
# R = 1

# Q_ = np.array([
#     [10, 0,  0, 0],
#     [ 0, 1,  0, 0],
#     [ 0, 0, 10, 0],
#     [ 0, 0,  0, 1]
# ], dtype=np.float64)
# R = np.array([[0.1]], dtype=np.float64)

Q_ = np.eye(4)
R = 0.0001

w_r = np.array([
    [1],
    [0],
    [np.pi],
    [0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat # (xs.shape[1], us.shape[1])
def reward(x, u):
    return -cost(x, u)

#%% Policy function as PyTorch model
class PolicyNetwork():
    def __init__(self, n_state, n_action, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state, n_action),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        """
            Compute the action probabilities of state s using the learning model
            @param s: input state array
            @return: predicted policy
        """
        return self.model(torch.Tensor(s))

    def update(self, returns, log_probs):
        """
            Update the weights of the policy network given the training samples
            @param returns: return (cumulative rewards) for each step in an episode
            @param log_probs: log probability for each step
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        """
            Estimate the policy and sample an action, compute its log probability
            @param s: input state
            @return: the selected action and log probability
        """
        probs = self.predict(s)
        action_index = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action_index])
        action = all_us[action_index].item()
        return action, log_prob

def reinforce(estimator, n_episode, gamma=1.0):
    """
        REINFORCE algorithm
        @param estimator: policy network
        @param n_episode: number of episodes
        @param gamma: the discount factor
    """
    w_hat = np.zeros([15,1])
    for episode in range(n_episode):
        states = []
        actions = []
        log_probs = []
        rewards = []
        perturbation = np.array([
            [0],
            [0],
            [np.random.normal(0, 0.05)],
            [0]
        ])
        state = x0 + perturbation
        # state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))

        while len(rewards) < 30:
            u, log_prob = estimator.get_action(state[:,0])
            action = np.array([[u]])
            next_state = f(state, action)
            curr_reward = reward(state, action)[0,0]

            states.append(state)
            actions.append(u)

            total_reward_episode[episode] += curr_reward
            log_probs.append(log_prob)
            rewards.append(curr_reward)

            if len(rewards) == 30:
                returns = torch.zeros([len(rewards)])
                # Gt = 0
                for i in range(len(rewards)-1, -1, -1):
                    # Gt = rewards[i] + (gamma * Gt)
                    # returns[i] = Gt
                    # print("Gt:", Gt)
                    Q_val = Q(
                        np.vstack(states[i]),
                        np.array([[actions[i]]]),
                        w_hat
                    )
                    returns[i] = Q_val
                    # print("Q_val:", Q_val)

                # if episode == 0 or (episode+1) % 100 == 0:
                #     print(returns)

                returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

                estimator.update(returns, log_probs)
                if episode == 0 or (episode+1) % 100 == 0:
                    print(f"Episode: {episode+1}, total reward: {total_reward_episode[episode]}")
                    # torch.save(estimator, 'lqr-policy-model.pt')

                break

            state = next_state

        w_hat = w_hat_t()

#%%
# n_state = env.observation_space.shape[0]
n_state = A.shape[0]
n_action = all_us.shape[0]
lr = 0.003
policy_net = PolicyNetwork(n_state, n_action, lr)
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)
policy_net.model.apply(init_weights)
# policy_net = torch.load('lqr-policy-model.pt')

n_episode = 2000
gamma = 0.99
lamb = 0.0001
total_reward_episode = [0] * n_episode

#%% Construct snapshots of data
# N = 20000
# state_range = 20
# action_range = 20
# X = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
# U = np.random.rand(1,N)*action_range*np.random.choice(np.array([-1,1]), size=(1,N))
# Y = f(X, U)

num_episodes = 100
num_steps_per_episode = 200

X = np.zeros([A.shape[0], num_episodes*num_steps_per_episode])
Y = np.zeros([A.shape[0], num_episodes*num_steps_per_episode])
U = np.zeros([1, num_episodes*num_steps_per_episode])

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
        u = np.array([[np.random.choice(all_us)]])
        U[:, (episode*num_steps_per_episode)+step] = u[:, 0]
        y = f(x, u)
        Y[:, (episode*num_steps_per_episode)+step] = y[:, 0]
        x = np.vstack(y)

#%% Estimate Koopman tensor
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
w_hat_batch_size = 2**10 # 2**14
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy_net.predict(x_batch.T).T # (all_us.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_us.data.numpy()])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response.data.numpy()) # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = np.einsum('wu,uw->wu', reward(x_batch, np.array([all_us.data.numpy()])), pi_response.data.numpy()) # (w_hat_batch_size, all_us.shape[0])
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

#%% REINFORCE
reinforce(policy_net, n_episode, gamma)

# import matplotlib.pyplot as plt
# plt.plot(total_reward_episode)
# plt.title('Episode reward over time')
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()

#%% Test policy
num_episodes = 1
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q_, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)

test_steps = 200
def watch_agent():
    optimal_states = np.zeros([test_steps,n_state])
    learned_states = np.zeros([test_steps,n_state])
    for episode in range(num_episodes):
        perturbation = np.array([
            [0],
            [0],
            [np.random.normal(0, 0.05)],
            [0]
        ])
        state = x0 + perturbation
        # state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
        optimal_state = state
        learned_state = state
        step = 0
        while step < test_steps:
            optimal_states[step] = optimal_state[:,0]
            learned_states[step] = learned_state[:,0]

            optimal_action = np.random.normal(-(C @ (optimal_state - w_r)), sigma_t)
            optimal_state = f(optimal_state, optimal_action)

            with torch.no_grad():
                learned_u, _ = policy_net.get_action(learned_state[:,0])
            learned_action = np.array([[learned_u]])
            learned_state = f(learned_state, learned_action)

            step += 1
    plt.plot(learned_states[:,0])
    plt.plot(learned_states[:,1])
    plt.plot(learned_states[:,2])
    plt.plot(learned_states[:,3])
    plt.show()
    plt.plot(optimal_states[:,0])
    plt.plot(optimal_states[:,1])
    plt.plot(optimal_states[:,2])
    plt.plot(optimal_states[:,3])
    plt.show()
watch_agent()

#%%