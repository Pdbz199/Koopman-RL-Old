import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make('CartPole-v1')
# env = gym.make('LunarLander-v2')

# seed = 1234
seed = 123
# env.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

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
        state_minimums=None,
        state_maximums=None,
        cost=None,
        save_data_path=None,
        gamma=0.99,
        learning_rate=0.01,
        is_gym_env=False,
        render=False,
        load_model=False,
        seed=123
    ):
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = env
        self.all_actions = all_actions
        self.dynamics_model = dynamics_model
        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.cost = cost
        self.save_data_path = save_data_path
        self.training_data_path = f"{self.save_data_path}/training_data"
        self.actor_critic_file_name = "ppo_policy.pt"
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.is_gym_env = is_gym_env
        self.render = render

        if load_model:
            self.actor_critic = torch.load(f"{self.save_data_path}/{self.actor_critic_file_name}")

            print("Loaded PyTorch model...")
        else:
            self.actor_critic = ActorCritic(
                input_dim=env.observation_space.shape[0] if is_gym_env else self.dynamics_model.X.shape[0],
                layer_1_dim=128,
                layer_2_dim=256,
                output_dim=env.action_space.n if is_gym_env else all_actions.shape[1],
            )

            # Initialize network weights to 0
            def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        m.weight.data.fill_(0.0)

            self.actor_critic.apply(init_weights)

            print("Initialized new PyTorch model...")

        # Initialize Actor-Critic optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        # PyTorch network friendly state
        torch_state = torch.FloatTensor(state[:, 0]).unsqueeze(0)

        # Don't compute gradients for inference
        with torch.no_grad():
            action_probabilities, _ = self.actor_critic(torch_state)

        # Construct action distribution, sample action, and get log probability
        action_distribution = distributions.Categorical(action_probabilities)
        action_index = action_distribution.sample()
        log_prob = action_distribution.log_prob(action_index)
        action = np.array([self.all_actions[:, action_index]])

        return action, log_prob

    def update_policy(
        self,
        states,
        action_indices,
        log_probs,
        entropies,
        advantages,
        returns,
        ppo_steps,
        ppo_clip
    ):
        """
            Compute the actor-critic loss and update the network weights.
        """

        action_indices = action_indices.detach()
        log_probs = log_probs.detach()
        entropies = entropies.detach()
        advantages = advantages.detach()

        # Update the policy gradients `ppo_steps` times
        for _ in range(ppo_steps):
            # Get latest action probabilities and V_x from the ActorCritic module 
            action_probabilities, V_x = self.actor_critic(states)
            action_distribution = distributions.Categorical(action_probabilities)
            V_x = V_x.squeeze(-1)

            # Compute new log probabilities of actions
            new_log_probs = action_distribution.log_prob(action_indices)

            # Compute policy ratio as in the PPO objective
            # Do subtraction instead of division because we are in log space
            policy_ratio = (new_log_probs - log_probs).exp()

            # Compute the two policy losses that we will take the minimum of as in PPO objective
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0-ppo_clip, max=1.0+ppo_clip) * advantages

            # Compute actor-critic loss
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = 0.5 * F.mse_loss(returns, V_x)
            entropy_loss = 0.01 * entropies.mean()
            actor_critic_loss = policy_loss + value_loss - entropy_loss

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
            action_indices = []
            actions = []
            log_probs = []
            entropies = []
            V_xs = []
            rewards = []
            done = False
            episode_reward = 0

            # Pick an initial state from the environment
            if self.is_gym_env:
                state = np.vstack(self.env.reset())
            else:
                state = np.random.uniform(
                    self.state_minimums,
                    self.state_maximums,
                    (self.dynamics_model.x_dim, 1)
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
                entropy = action_distribution.entropy()

                # Push the environment one step forward using the sampled action
                if self.is_gym_env:
                    action = action_index.item() # Extract numerical value from object
                    state, reward, done, _ = self.env.step(action)
                    state = np.vstack(state) # Column vector

                    if self.render:
                        self.env.render()
                else:
                    action = np.array([self.all_actions[:, action_index]])
                    reward = -self.cost(state, action)[0, 0]
                    state = self.env(state, action)
                    done = step_num == num_steps_per_episode

                # Add to running episode reward
                episode_reward += reward

                # Track data over time
                action_indices.append(action_index)
                actions.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)
                V_xs.append(V_x)
                rewards.append(reward)

                step_num += 1

            states = torch.cat(states)
            action_indices = torch.cat(action_indices)
            actions = torch.FloatTensor(np.array(actions))
            log_probs = torch.cat(log_probs)
            entropies = torch.cat(entropies)
            V_xs = torch.cat(V_xs).squeeze(-1)

            # Calculate returns and advantages
            returns = calculate_returns(rewards, self.gamma)
            advantages = calculate_advantages(returns, V_xs)

            # Update network weights and output the loss
            actor_critic_loss = self.update_policy(
                states,
                action_indices,
                log_probs,
                entropies,
                advantages,
                returns,
                ppo_steps,
                ppo_clip
            )
            print(actor_critic_loss)
            training_losses.append(actor_critic_loss)

            # Track training rewards
            training_rewards.append(rewards)

            # Compute the mean training reward over the last `num_trials` episodes
            mean_training_rewards = np.mean(np.sum(training_rewards[-num_trials:], axis=1))

            # Log progress every so often
            if episode_num == 1 or episode_num % print_every == 0:
                print(f"| Episode: {episode_num:4} | Episode Reward: {episode_reward:5.1f} | Mean Train Rewards: {mean_training_rewards:5.1f} |")

                # Save models
                if not self.is_gym_env:
                    torch.save(self.actor_critic, f"{self.save_data_path}/{self.actor_critic_file_name}")

                # Save training data
                # np.save(f"{self.training_data_path}/v_xs.npy", torch.Tensor(V_xs).detach().numpy())
                # np.save(f"{self.training_data_path}/v_x_primes.npy", torch.Tensor(V_x_primes).detach().numpy())
                # np.save(f"{self.training_data_path}/actions.npy", np.array(actions))
                # np.save(f"{self.training_data_path}/log_probs.npy", torch.Tensor(log_probs).detach().numpy())
                np.save(f"{self.training_data_path}/training_losses.npy", np.array(training_losses))
                np.save(f"{self.training_data_path}/rewards.npy", np.array(training_rewards))

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
        save_data_path="./analysis/tmp/proximal_policy_optimization",
        learning_rate=0.001,
        gamma=0.99,
        is_gym_env=True,
        render=False,
        seed=seed
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