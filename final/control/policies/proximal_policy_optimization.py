import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

# env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v2')

seed = 1234
# seed = 123
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

epsilon = np.finfo(np.float32).eps.item()

def calculate_returns(rewards, discount_factor, normalize=True):
    """
        Calculate returns given earned rewards and a discount factor.
    """

    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + discount_factor*R
        returns.insert(0, R)

    returns = torch.FloatTensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

    return returns

def calculate_advantages(returns, V_xs, normalize=True):
    """
        Calculate advantages given returns and estimated V_xs.
    """

    advantages = returns - V_xs

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + epsilon)

    return advantages

class ActorCritic(nn.Module):
    def __init__(self, input_dim, layer_1_dim, layer_2_dim, output_dim):
        super(ActorCritic, self).__init__()

        # Policy network is a multi-layer perceptron with 2 hidden layers
        self.policy_layer_1 = nn.Linear(input_dim, layer_1_dim)
        self.policy_layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.policy_output_layer = nn.Linear(layer_2_dim, output_dim)

        # Value network is a multi-layer perceptron with 2 hidden layer
        self.value_layer_1 = nn.Linear(input_dim, layer_1_dim)
        self.value_layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.value_output_layer = nn.Linear(layer_2_dim, 1)

    def forward(self, x):
        # Compute policy network output
        _x = self.policy_layer_1(x)
        _x = F.dropout(_x)
        _x = F.relu(_x)
        _x = F.relu(self.policy_layer_2(_x))
        pi = F.softmax(self.policy_output_layer(_x), dim=-1)

        # Compute value network output
        _x = self.value_layer_1(x)
        _x = F.dropout(_x)
        _x = F.relu(_x)
        _x = F.relu(self.value_layer_2(_x))
        v = self.value_output_layer(_x)

        return pi, v

class ProximalPolicyOptimization:
    def __init__(
        self,
        env,
        all_actions=None,
        dynamics_model=None,
        cost=None,
        gamma=0.99,
        learning_rate=0.01,
        is_gym_env=False,
        render=False,
    ):
        self.env = env
        self.all_actions = all_actions
        self.dynamics_model = dynamics_model
        self.cost = cost
        self.actor_critic = ActorCritic(
            input_dim=env.observation_space.shape[0],
            layer_1_dim=128,
            layer_2_dim=256,
            output_dim=env.action_space.n if is_gym_env else all_actions.shape[1],
        )
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.is_gym_env = is_gym_env
        self.render = render

        # Initialize network weights to 0
        def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)

        self.actor_critic.apply(init_weights)

        # Initialize Actor-Critic optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

    def update_policy(
        self,
        states,
        actions,
        log_probs,
        advantages,
        returns,
        ppo_steps,
        ppo_clip
    ):
        """
            Compute the actor-critic loss and update the network weights.
        """

        advantages = advantages.detach()
        log_probs = log_probs.detach()
        actions = actions.detach()

        # Update the policy gradients `ppo_steps` times
        for _ in range(ppo_steps):
            # Get latest action probabilities and V_x from the ActorCritic module 
            action_probabilities, V_x = self.actor_critic(states)
            action_distribution = distributions.Categorical(action_probabilities)
            V_x = V_x.squeeze(-1)

            # Compute new log probabilities of actions
            new_log_probs = action_distribution.log_prob(actions)

            # Compute policy ratio as in the PPO objective
            # Do subtraction instead of division because we are in log space
            policy_ratio = (new_log_probs - log_probs).exp()

            # Compute the two policy losses that we will take the minimum of as in PPO objective
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0-ppo_clip, max=1.0+ppo_clip) * advantages

            # Compute actor-critic loss
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.mse_loss(returns, V_x)
            actor_critic_loss = policy_loss + value_loss

            # Zero out gradients
            self.optimizer.zero_grad()

            # Backpropagation
            actor_critic_loss.backward()
            self.optimizer.step()

        # Return the loss value
        return actor_critic_loss.item()

    def train(
        self,
        num_episodes,
        num_trials,
        ppo_steps,
        ppo_clip,
        reward_threshold,
        print_every,
        num_steps_per_episode=200
    ):
        training_rewards = []
        training_losses = []

        for episode_num in range(1, num_episodes+1):
            states = []
            actions = []
            log_probs = []
            V_xs = []
            rewards = []
            done = False
            episode_reward = 0

            # Pick an initial state from the environment
            if self.is_gym_env:
                state = np.vstack(self.env.reset())
            else:
                state = np.vstack(
                    np.random.uniform(
                        self.state_minimums,
                        self.state_maximums,
                        self.dynamics_model.x_dim
                    )
                )

            step_num = 1
            while not done:
                # PyTorch network friendly state
                torch_state = torch.FloatTensor(state[:, 0]).unsqueeze(0)
                states.append(torch_state)

                # Get action probabilities and V_x from the ActorCritic module and sample action index
                action_probabilities, V_x = self.actor_critic(torch_state)
                action_distribution = distributions.Categorical(action_probabilities)
                action_index = action_distribution.sample()
                log_prob = action_distribution.log_prob(action_index)

                # Push the environment one step forward using the sampled action
                if self.is_gym_env:
                    action = action_index.item() # Extract numerical value from object
                    state, reward, done, _ = self.env.step(action)
                    state = np.vstack(state) # Column vector

                    if self.render:
                        self.env.render()
                else:
                    action = self.all_actions[:, action_index]
                    reward = -self.cost(state, action)
                    state = self.env(state, action)
                    done = step_num == num_steps_per_episode

                # Add to running episode reward
                episode_reward += reward

                # Track data over time
                actions.append(action)
                log_probs.append(log_prob)
                V_xs.append(V_x)
                rewards.append(reward)

                step_num += 1

            states = torch.cat(states)
            # actions = torch.cat(actions)
            actions = torch.FloatTensor(actions)
            log_probs = torch.cat(log_probs)
            V_xs = torch.cat(V_xs).squeeze(-1)

            # Calculate returns and advantages
            returns = calculate_returns(rewards, self.gamma)
            advantages = calculate_advantages(returns, V_xs)

            # Update network weights and output the loss
            actor_critic_loss = self.update_policy(
                states,
                actions,
                log_probs,
                advantages,
                returns,
                ppo_steps,
                ppo_clip
            )
            training_losses.append(actor_critic_loss)

            # Track training rewards
            training_rewards.append(episode_reward)

            # Compute the mean training reward over the last `num_trials` episodes
            mean_training_rewards = np.mean(training_rewards[-num_trials:])

            # Log progress every so often
            if episode_num % print_every == 0:
                print(f"| Episode: {episode_num:3} | Episode Reward: {episode_reward:5.1f} | Mean Train Rewards: {mean_training_rewards:5.1f} |")

            # If we surpass reward threshold, stop training
            if mean_training_rewards >= reward_threshold:
                print(f"Reached reward threshold in {episode_num} episodes")
                break

        # Return training rewards and losses
        return training_rewards, training_losses





if __name__ == "__main__":
    num_episodes = 5000
    num_trials = 25
    ppo_steps = 10
    ppo_clip = 0.2
    # reward_threshold = 475 # CartPole-v1
    reward_threshold = 200 # LunarLandar-v2
    print_every = 10

    ppo = ProximalPolicyOptimization(
        env,
        learning_rate=0.001,
        gamma=0.99,
        is_gym_env=True,
        render=False
    )
    print("Actor-Critic Model:\n", ppo.actor_critic)
    training_rewards, training_losses = ppo.train(
        num_episodes,
        num_trials,
        ppo_steps,
        ppo_clip,
        reward_threshold,
        print_every
    )

    fig = plt.figure(figsize=(12,8))

    # Calculate the linear regressions
    xs = np.arange(len(training_rewards))

    training_rewards_slope, training_rewards_intercept = np.polyfit(xs, training_rewards, 1)
    training_rewards_line = training_rewards_slope * xs + training_rewards_intercept

    training_losses_slope, training_losses_intercept = np.polyfit(xs, training_losses, 1)
    training_losses_line = training_losses_slope * xs + training_losses_intercept

    ax = fig.add_subplot(2, 1, 1)
    ax.set_title(f'Training Rewards (Linear Fit Slope = {training_rewards_slope})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(training_rewards, label='Training Reward')
    ax.plot(training_rewards_line, label='Linear Fit')
    ax.hlines(reward_threshold, 0, len(training_rewards), color='r', label='Reward Threshold')
    ax.legend(loc='lower right')

    ax = fig.add_subplot(2, 1, 2)
    ax.set_title(f'Training Loss (Linear Fit Slope = {training_losses_slope})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.plot(training_losses, label='Training Loss')
    ax.plot(training_losses_line, label='Linear Fit')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()