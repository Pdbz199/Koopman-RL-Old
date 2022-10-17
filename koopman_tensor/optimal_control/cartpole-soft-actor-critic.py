import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pybullet_envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

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


# if __name__ == '__main__':
#     if os.path.exists('model/actor.pkl'):
#         actor = torch.load('model/actor.pkl')
#         print('Actor Model loaded')
#     else:
#         actor = Actor(state_size, action_size).to(device)
#     if os.path.exists('model/critic.pkl'):
#         critic = torch.load('model/critic.pkl')
#         print('Critic Model loaded')
#     else:
#         critic = Critic(state_size, action_size).to(device)
#     trainIters(actor, critic, num_episodes=2000, num_steps_per_episode=50)


""""""""""""""""""""""""""" SOFT ACTOR CRITIC IMPLEMENTATION """""""""""""""""""""""""""

#%%
class ReplayBuffer():
    def __init__(self, max_size, input_shape, num_actions):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = torch.zeros([self.memory_size, *input_shape])
        self.new_state_memory = torch.zeros([self.memory_size, *input_shape])
        self.action_memory = torch.zeros([self.memory_size, num_actions])
        self.reward_memory = torch.zeros([self.memory_size])
        self.terminal_memory = torch.zeros(self.memory_size, dtype=torch.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = torch.tensor(state)
        self.new_state_memory[index] = torch.tensor(state_)
        self.action_memory[index] = torch.tensor(action)
        self.reward_memory[index] = torch.tensor(reward)
        self.terminal_memory[index] = torch.tensor(done)

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.memory_size)

        batch_indices = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]

        return states, states_, actions, rewards, dones

#%%
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dimensions, num_actions, fc1_dimensions=256, fc2_dimensions=256, name='critic', checkpoint_directory='tmp/sac'):
        super(CriticNetwork, self).__init__()

        self.input_dimensions = input_dimensions
        self.num_actions = num_actions
        self.name = name
        self.fc1_dimensions = fc1_dimensions
        self.fc2_dimensions = fc2_dimensions
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_directory, self.name+'_sac')

        self.fc1 = nn.Linear(self.input_dimensions[0] + self.num_actions, self.fc1_dimensions)
        self.fc2 = nn.Linear(self.fc1_dimensions, self.fc2_dimensions)
        self.q = nn.Linear(self.fc2_dimensions, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        # torch.save(self.state_dict(), self.checkpoint_file)
        pass

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dimensions, fc1_dimensions=256, fc2_dimensions=256, name='value', checkpoint_directory='tmp/sac'):
        super(ValueNetwork, self).__init__()

        self.input_dimensions = input_dimensions
        self.fc1_dimensions = fc1_dimensions
        self.fc2_dimensions = fc2_dimensions
        self.name = name
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_directory, self.name+'_sac')

        self.fc1 = nn.Linear(*self.input_dimensions, self.fc1_dimensions)
        self.fc2 = nn.Linear(self.fc1_dimensions, self.fc2_dimensions)
        self.v = nn.Linear(self.fc2_dimensions, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dimensions, max_action, num_actions=2, fc1_dimensions=256, fc2_dimensions=256, name='actor', checkpoint_directory='tmp/sac'):
        super(ActorNetwork, self).__init__()

        self.input_dimensions = input_dimensions
        self.max_action = max_action
        self.num_actions = num_actions
        self.fc1_dimensions = fc1_dimensions
        self.fc2_dimensions = fc2_dimensions
        self.reparam_noise = 1e-6
        self.name = name
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_directory, self.name+'_sac')

        self.fc1 = nn.Linear(*self.input_dimensions, self.fc1_dimensions)
        self.fc2 = nn.Linear(self.fc1_dimensions, self.fc2_dimensions)
        self.mu = nn.Linear(self.fc2_dimensions, self.num_actions)
        self.sigma = nn.Linear(self.fc2_dimensions, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

#%% Agent
class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dimensions=[8], env=None, gamma=0.99, num_actions=2, max_size=1_000_000, tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dimensions, num_actions)
        self.batch_size = batch_size
        self.num_actions = num_actions

        self.actor = ActorNetwork(alpha, input_dimensions, max_action=env.action_space.high, num_actions=num_actions)
        self.critic_1 = CriticNetwork(beta, input_dimensions, num_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dimensions, num_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dimensions)
        self.target_value = ValueNetwork(beta, input_dimensions, name='target_value')

        self.reward_scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.Tensor(observation).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

            self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('..... saving models .....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('..... loading models .....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        #state, action, reward, new_state, done = \
        state, new_state, action, reward, done = \
                self.memory.sample_buffer(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)

        value = self.value(state)#.view(-1)
        value_ = self.target_value(state_)#.view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        # log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        # critic_value = critic_value.view(-1)

        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        # log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        # critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        q_hat = self.reward_scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action)#.view(-1)
        q2_old_policy = self.critic_2.forward(state, action)#.view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    # env = gym.make('CartPole-v0')
    # env = gym.make('Pendulum-v1')
    # env = gym.make('MountainCarContinuous-v0')
    agent = Agent(input_dimensions=env.observation_space.shape, env=env, num_actions=env.action_space.shape[0])
    num_games = 150
    # num_games = 0

    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    # load_checkpoint = True

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for game in range(num_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = [agent.choose_action(observation)]
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f"episode {game}, score {score}, avg_score {avg_score}")

    if not load_checkpoint:
        x = [i+1 for i in range(num_games)]
        plot_learning_curve(x, score_history, figure_file)