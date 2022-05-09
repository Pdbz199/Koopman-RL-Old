'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 8: Implementing Policy Gradients and Policy Optimization
Author: Yuxi (Hayden) Liu
'''

import torch
import gym
import torch.nn as nn

# env_string = 'CartPole-v0'
env_string = 'env:CartPoleControlEnv-v0'
env = gym.make(env_string)

step_size = 1.0 if env_string == 'CartPole-v0' else 0.5
all_us = torch.arange(0, 1+step_size, step_size) if env_string == 'CartPole-v0' else torch.arange(-10, 10+step_size, step_size)

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
        if env_string == 'CartPole-v0':
            action = int(action)
        return action, log_prob

def reinforce(env, estimator, n_episode, gamma=1.0):
    """
        REINFORCE algorithm
        @param env: Gym environment
        @param estimator: policy network
        @param n_episode: number of episodes
        @param gamma: the discount factor
    """
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state = env.reset()

        while len(rewards) < 1000:
            action, log_prob = estimator.get_action(state)
            next_state, reward, is_done, _ = env.step(action)

            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)

            if is_done:
                returns = torch.zeros([len(rewards)])
                Gt = 0
                for i in range(len(rewards)-1, -1, -1):
                    Gt = rewards[i] + (gamma * Gt)
                    returns[i] = Gt

                returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

                estimator.update(returns, log_probs)
                if episode % 500 == 0:
                    print(f"Episode: {episode}, total {'reward' if env_string == 'CartPole-v0' else 'cost'}: {total_reward_episode[episode]}")

                break

            state = next_state

n_state = env.observation_space.shape[0]
n_action = all_us.shape[0]
lr = 0.003
policy_net = PolicyNetwork(n_state, n_action, lr)

n_episode = 6000 # Completely converged at 2000 episodes for original code
gamma = 0.99
total_reward_episode = [0] * n_episode

reinforce(env, policy_net, n_episode, gamma)

# import matplotlib.pyplot as plt
# plt.plot(total_reward_episode)
# plt.title('Episode reward over time')
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()

#%% Test in environment
num_episodes = 10
def watch_agent():
    rewards = torch.zeros([num_episodes])
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            env.render()
            with torch.no_grad():
                action, _ = policy_net.get_action(state)
            state, _, done, __ = env.step(action)
            step += 1
            if done or step >= 200:
                rewards[episode] = step
                print("Reward:", step)
    env.close()
    print(f"Mean reward per episode over {num_episodes} episodes:", torch.mean(rewards))
watch_agent()