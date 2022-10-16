import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

#%% Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = gym.make("CartPole-v0").unwrapped

# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n
# lr = 0.0001
state_size = 3
action_size = 1
lr = 0.003

#%% Dynamics
A = np.zeros([state_size, state_size])
max_abs_real_eigen_val = 1.0
while max_abs_real_eigen_val >= 1.0 or max_abs_real_eigen_val <= 0.7:
    Z = np.random.rand(*A.shape)
    _,sigma,__ = np.linalg.svd(Z)
    Z /= np.max(sigma)
    A = Z.T @ Z
    W,_ = np.linalg.eig(A)
    max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

print("A:", A)
print("A's max absolute real eigenvalue:", max_abs_real_eigen_val)
B = np.ones([state_size,action_size])

def f(x, u):
    return A @ x + B @ u

#%% Define cost
# Q = np.eye(state_size)
Q = torch.eye(state_size)
R = 1
w_r = np.array([
    [0.0],
    [0.0],
    [0.0]
])
# def cost(x, u):
#     # Assuming that data matrices are passed in for X and U. Columns are snapshots
#     # x.T Q x + u.T R u
#     x_ = x - w_r
#     mat = np.vstack(np.diag(x_.T @ Q @ x_)) + np.power(u, 2)*R
#     return mat.T
def cost(x, u):
    return x.T @ Q @ x + u * R * u

#%% Neural networks
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R # * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, num_episodes, num_steps_per_episode):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())

    # state_range = 25.0
    state_range = 5.0
    state_minimums = np.ones([state_size,1]) * -state_range
    state_maximums = np.ones([state_size,1]) * state_range

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [state_size, num_episodes]
    )

    for episode in range(num_episodes):
        # state = env.reset()
        state = np.vstack(initial_states[:,episode])
        states = [state]
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for step in range(num_steps_per_episode):
            state_tensor = torch.FloatTensor(state[:,0]).to(device)
            dist, value = actor(state_tensor), critic(state_tensor)

            u = dist.sample()
            action = np.array([[u]])
            log_prob = dist.log_prob(u).unsqueeze(0)
            entropy += dist.entropy().mean()

            reward = -cost(state_tensor, u)

            state = f(state, action)

            states.append(state)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float, device=device))

        states = np.array(states)

        print(f"Iteration: {episode+1}, Reward: {np.sum(rewards)}")
        if (episode+1) % 250 == 0:
            plt.plot(states.reshape([len(states),state_size]))
            plt.show()

        next_state = torch.FloatTensor(state[:,0]).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    # torch.save(actor, 'model/actor.pkl')
    # torch.save(critic, 'model/critic.pkl')
    # env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, num_episodes=2000, num_steps_per_episode=50)