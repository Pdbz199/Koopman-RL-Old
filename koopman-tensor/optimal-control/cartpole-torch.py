import gym
import numpy as np
import torch

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

from matplotlib import pyplot as plt
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
# import cartpole_reward
import observables

env = gym.make('CartPole-v0')

#%% System dynamics
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
# R = 0.1

# def f(x, u):
#     return A @ x + B @ u

#%% Construct snapshots of u from random agent and initial states x0
N = 20000
# action_range = 25
# state_range = 25
# U = np.random.rand(1,N)*action_range*np.random.choice(np.array([-1,1]), size=(1,N))
U = np.zeros([1,N])
# X0 = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
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
Ns = np.arange( state_dim - 1, state_dim - 1 + (order+1) )
obs_size = int( np.sum( comb( Ns, np.ones_like(Ns) * (order+1) ) ) )
n_actions = env.action_space.n
# HIDDEN_SIZE = 256

# model = torch.nn.Sequential(
#     torch.nn.Linear(obs_size, HIDDEN_SIZE),
#     torch.nn.ReLU(),
#     torch.nn.Linear(HIDDEN_SIZE, n_actions),
#     torch.nn.Softmax(dim=0)
# )
model = torch.nn.Sequential(
    torch.nn.Linear(obs_size, n_actions),
    torch.nn.Softmax(dim=0)
)
print("Model:", model)

# learning_rate = 0.003
learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Horizon = 500
MAX_TRAJECTORIES = 5000 # 500, 750
gamma = 0.99
score = []

for trajectory in range(MAX_TRAJECTORIES):
    curr_state = np.vstack(env.reset()) # column vector
    done = False
    transitions = []

    for t in range(Horizon):
        phi_curr_state = tensor.phi(curr_state) # column vector
        act_prob = model(torch.from_numpy(phi_curr_state[:,0]).float())
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
        prev_state = curr_state[:,0]
        # curr_state, _, done, info = env.step(action)
        curr_state = tensor.B.T @ tensor.K_(np.array([[action]])) @ phi_curr_state
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

    if trajectory % 50 == 0 and trajectory>0:
        print('Trajectory {}\tAverage Score: {:.2f}'
                .format(trajectory, np.mean(score[-50:-1])))

def running_mean(x):
    N=50
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