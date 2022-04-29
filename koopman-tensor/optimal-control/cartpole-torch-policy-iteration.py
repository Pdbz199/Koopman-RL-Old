import gym
import numpy as np
import random
import torch

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

from matplotlib import pyplot as plt
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import cartpole_reward
import estimate_L
import observables

#%% Initialize environment
env = gym.make('CartPole-v0')
def cost(xs, us):
    return -cartpole_reward.defaultCartpoleRewardMatrix(xs, us)

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
        Y[:,i], _, done, __ = env.step(int(U[0,i]))
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
    psi=observables.monomials(order),
    regressor='ols'
)

# obs_size = env.observation_space.shape[0]
state_dim = env.observation_space.shape[0]
M_plus_N_minus_ones = np.arange( (state_dim-1), order + (state_dim-1) + 1 )
phi_dim = int( np.sum( comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * (state_dim-1) ) ) )
action_dim = env.action_space.n
u_range = np.array([0, 2])
all_us = np.arange(u_range[0], u_range[1])

# HIDDEN_SIZE = 256

# model = torch.nn.Sequential(
#     torch.nn.Linear(obs_size, HIDDEN_SIZE),
#     torch.nn.ReLU(),
#     torch.nn.Linear(HIDDEN_SIZE, n_actions),
#     torch.nn.Softmax(dim=0)
# )
model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, action_dim),
    torch.nn.Softmax(dim=0)
)
# print("Model:", model)

learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Horizon = 500
MAX_TRAJECTORIES = 4000 # 500, 750 with other NN spec
gamma = 0.99
score = []

def w_hat_t(x):
    phi_x = tensor.phi(x)
    pi_response = model(torch.from_numpy(phi_x).float())
    phi_x_primes = np.zeros_like(all_us)
    costs = np.zeros_like(all_us)
    for i in range(all_us.shape[0]):
        phi_x_primes[i] = tensor.K_(all_us[i]) @ phi_x * pi_response[i]
        costs[i] = cost(x, all_us[i]) * pi_response[i]
    expectation_term_1 = torch.mean(phi_x_primes)
    expectation_term_2 = torch.mean(costs)

    # torch.sum(
    #     torch.pow(
    #         (w.T @ (phi_x - expectation_term_1)) - expectation_term_2,
    #         2
    #     )
    # )

    # || Y      - XB  ||
    # || B^TX^T - Y^T ||
    w_hat = estimate_L.ols(
        (phi_x - expectation_term_1).T,
        expectation_term_2.T
    ).T

    return w_hat

def Q(x, u):
    return cost(x, u) + w_hat_t(x).T @ tensor.phi_f(x, u)

for trajectory in range(MAX_TRAJECTORIES):
    curr_state = np.vstack(env.reset())
    done = False
    transitions = []

    for t in range(Horizon):
        phi_curr_state = tensor.phi(curr_state)
        act_prob = model(torch.from_numpy(phi_curr_state[:,0]).float())
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
        prev_state = curr_state[:,0]
        curr_state, _, done, info = env.step(action)
        curr_state = np.vstack(curr_state)
        done = bool(
            curr_state[0,0] < -env.x_threshold
            or curr_state[0,0] > env.x_threshold
            or curr_state[2,0] < -env.theta_threshold_radians
            or curr_state[2,0] > env.theta_threshold_radians
        )
        transitions.append((prev_state, action, t+1))
        if done:
            break
    score.append(len(transitions))
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))

    batch_Gvals = []
    for i in range(len(transitions)):
        new_Gval = 0
        power = 0
        for j in range(i, len(transitions)):
            new_Gval = new_Gval + ((gamma**power) * reward_batch[j]).numpy()
            power += 1
        batch_Gvals.append(new_Gval)
    expected_returns_batch = torch.FloatTensor(batch_Gvals)
    expected_returns_batch /= expected_returns_batch.max()

    state_batch = torch.Tensor(tensor.phi(np.array([s for (s,a,r) in transitions]).T).T)
    action_batch = torch.Tensor([a for (s,a,r) in transitions])

    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()

    loss = -torch.sum(torch.log(prob_batch) * expected_returns_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Online learning
    X = np.append(X, tensor.B.T @ state_batch.T.data.numpy(), axis=1)
    Y = np.roll(X, -1)[:,:-1]
    X = X[:,:-1]
    U = np.append(U, np.array([action_batch.data.numpy()]), axis=1)
    tensor = KoopmanTensor(
        X,
        Y,
        U,
        phi=observables.monomials(order),
        psi=observables.monomials(order),
        regressor='ols'
    )

    # Average score for trajectory
    if trajectory % 50 == 0 and trajectory>0:
        print(f'Trajectory {trajectory}\tAverage Score: {np.mean(score[-50:-1])}')

def running_mean(x):
    N = 50
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y
score = np.array(score)
avg_score = running_mean(score)
plt.figure(figsize=(15,7))
plt.ylabel("Trajectory Duration", fontsize=12)
plt.xlabel("Training Epochs", fontsize=12)
plt.plot(score, color='gray' , linewidth=1)
plt.plot(avg_score, color='blue', linewidth=3)
plt.scatter(np.arange(score.shape[0]), score, color='green' , linewidth=0.3)
# plt.show()

num_episodes = 5
def watch_agent():
    rewards = np.zeros([num_episodes])
    for episode in range(num_episodes):
        state = np.vstack(env.reset())
        done = False
        episode_rewards = []
        while not done:
            env.render()
            phi_state = tensor.phi(state)
            pred = model(torch.from_numpy(phi_state[:,0]).float())
            action = np.random.choice(np.array([0,1]), p=pred.data.numpy())
            state, reward, done, _ = env.step(action)
            state = np.vstack(state)
            episode_rewards.append(reward)
            if done:
                rewards[episode] = np.sum(episode_rewards)
                print("Reward:", rewards[episode])
    env.close()
    print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))
watch_agent()