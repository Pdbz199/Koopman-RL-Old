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
# env = gym.make(
#     'LunarLander-v2',
#     continuous=True
# )
# env = gym.make("ALE/Asteroids-v5")

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

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class SoftmaxPolicy(nn.Module):
    '''
    Simple neural network with softmax action selection
    '''
    def __init__(
        self,
        input_dim,
        hidden_layer_1_dim,
        hidden_layer_2_dim,
        num_actions
    ):
        super(SoftmaxPolicy, self).__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_layer_1_dim)
        self.linear_2 = nn.Linear(hidden_layer_1_dim, hidden_layer_2_dim)
        self.linear_3 = nn.Linear(hidden_layer_2_dim, num_actions)

    def forward(self, x):
        # x = F.dropout(x, p=0.2)
        x = self.linear_1(x)
        x = F.tanh(x)
        # x = F.relu(x)
        x = self.linear_2(x)
        x = F.tanh(x)
        # x = F.relu(x)
        x = self.linear_3(x)
        return F.softmax(x, dim=-1)

class GaussianPolicy(nn.Module):
    """
        Gaussian policy that consists of a neural network with 1 hidden layer that
        outputs mean and log std dev (the params) of a gaussian policy
    """

    def __init__(
        self,
        input_dim,
        hidden_layer_1_dim,
        hidden_layer_2_dim,
        action_dim
    ):
        super(GaussianPolicy, self).__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_layer_1_dim)
        self.linear_2 = nn.Linear(hidden_layer_1_dim, hidden_layer_2_dim)

        self.mean = nn.Linear(hidden_layer_2_dim, action_dim)
        self.log_std = nn.Linear(hidden_layer_2_dim, action_dim)

    def forward(self, x):
        # x = F.dropout(x, p=0.2)
        x = self.linear_1(x)
        x = F.tanh(x)
        # x = F.relu(x)
        x = self.linear_2(x)
        x = F.tanh(x)
        # x = F.relu(x)

        mean = self.mean(x)
        # If more than one action this will give you the diagonal elements of a diagonal covariance matrix
        log_std = self.log_std(x)
        # We limit the variance by forcing within a range of -2, 20
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        return mean, std

class ValueNetwork(nn.Module):
    """
        Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
        update. This a Neural Net with 1 hidden layer
    """

    def __init__(self, num_inputs, hidden_layer_1_dim, hidden_layer_2_dim):
        super(ValueNetwork, self).__init__()

        self.linear_1 = nn.Linear(num_inputs, hidden_layer_1_dim)
        self.linear_2 = nn.Linear(hidden_layer_1_dim, hidden_layer_2_dim)
        self.linear_3 = nn.Linear(hidden_layer_2_dim, 1)

    def forward(self, x):
        # x = F.dropout(x, p=0.2)
        x = self.linear_1(x)
        x = F.tanh(x)
        # x = F.relu(x)
        x = self.linear_2(x)
        x = F.tanh(x)
        # x = F.relu(x)
        x = self.linear_3(x)

        return x

class ActorCritic(nn.Module):
    def __init__(
        self,
        input_dim,
        layer_1_dim,
        layer_2_dim,
        output_dim,
        is_continuous=False
    ):
        super(ActorCritic, self).__init__()

        if is_continuous:
            self.policy = GaussianPolicy(
                input_dim,
                layer_1_dim,
                layer_2_dim,
                output_dim
            )
        else:
            self.policy = SoftmaxPolicy(
                input_dim,
                layer_1_dim,
                layer_2_dim,
                output_dim
            )

        self.value_function = ValueNetwork(
            input_dim,
            layer_1_dim,
            layer_2_dim
        )

    def forward(self, x):
        pi = self.policy(x)
        v_x = self.value_function(x)

        return pi, v_x

class ProximalPolicyOptimization:
    def __init__(
        self,
        env,
        all_actions=None,
        is_continuous=False,
        dynamics_model=None,
        state_minimums=None,
        state_maximums=None,
        cost=None,
        save_data_path=None,
        gamma=0.99,
        value_beta=1.0,
        entropy_beta=0.01,
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
        self.is_continuous = is_continuous
        self.dynamics_model = dynamics_model
        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.cost = cost
        self.save_data_path = save_data_path
        self.training_data_path = f"{self.save_data_path}/training_data"
        self.controller_type = "continuous" if self.is_continuous else "discrete"
        self.actor_critic_file_name = f"{self.controller_type}_ppo_policy.pt"
        self.gamma = gamma
        self.value_beta = value_beta
        self.entropy_beta = entropy_beta
        self.learning_rate = learning_rate
        self.is_gym_env = is_gym_env
        self.render = render

        if load_model:
            self.actor_critic = torch.load(f"{self.save_data_path}/{self.actor_critic_file_name}")

            print("\nLoaded PyTorch model...")
        else:
            if is_continuous:
                output_dim = 1
            elif is_gym_env:
                output_dim = env.action_space.n
            else:
                output_dim = all_actions.shape[1]
            self.actor_critic = ActorCritic(
                input_dim=env.observation_space.shape[0] if is_gym_env else self.dynamics_model.X.shape[0],
                layer_1_dim=64,
                layer_2_dim=64,
                output_dim=output_dim,
                is_continuous=is_continuous
            )

            # Initialize network weights to 0
            def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        m.weight.data.fill_(0.0)

            self.actor_critic.apply(init_weights)

            print("\nInitialized new PyTorch model...")

        # Initialize Actor-Critic optimizer and lr scheduler
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=1e-5)

    def get_action(self, state):
        """
            Function to get an action from the policy at inference time.
        """

        # PyTorch network friendly state
        torch_state = torch.FloatTensor(state[:, 0]).unsqueeze(0)

        # Get action probabilities from Actor-Critic module
        with torch.no_grad():
            # Construct action distribution, sample action, and get log probability
            if self.is_continuous:
                (action_mus, action_sigmas), _ = self.actor_critic(torch_state)
                action_distribution = distributions.Normal(action_mus, action_sigmas)
            else:
                action_probabilities, _ = self.actor_critic(torch_state)
                action_distribution = distributions.Categorical(action_probabilities)

            action_index = action_distribution.sample()

        if self.is_gym_env:
            action = action_index
        else:
            action = np.array([self.all_actions[:, action_index]])
        log_prob = action_distribution.log_prob(action_index)

        return action, log_prob

    def update_policy(
        self,
        states,
        actions,
        log_probs,
        entropies,
        advantages,
        returns,
        ppo_steps,
        ppo_clip,
        episode_num
    ):
        """
            Compute the actor-critic loss and update the network weights.
        """

        actions = actions.detach()
        log_probs = log_probs.detach()
        entropies = entropies.detach()
        advantages = advantages.detach()

        # Update the policy gradients `ppo_steps` times
        for _ in range(ppo_steps):
            if self.is_continuous:
                # Get latest action and V_x from the ActorCritic module 
                (action_mus, action_sigmas), V_x = self.actor_critic(states)
                action_distribution = distributions.Normal(action_mus, action_sigmas)
            else:
                # Get latest action probabilities and V_x from the ActorCritic module 
                action_probabilities, V_x = self.actor_critic(states)
                action_distribution = distributions.Categorical(action_probabilities)
            V_x = V_x.squeeze(-1)

            # Compute new log probabilities of actions
            new_log_probs = action_distribution.log_prob(actions)

            # Compute policy ratio as in the PPO objective
            # Do subtraction instead of division because we are in log space
            policy_ratio = (new_log_probs - log_probs).exp()

            # Compute the two policy losses that we will take the minimum of as in PPO objective (refer to eq. 7 in PPO paper)
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0-ppo_clip, max=1.0+ppo_clip) * advantages

            # Compute actor-critic loss (refer to eq. 9 in PPO paper)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.mse_loss(returns, V_x)
            entropy_loss = entropies.mean()
            actor_critic_loss = policy_loss + self.value_beta*value_loss
            ppo_loss = actor_critic_loss - self.entropy_beta*entropy_loss

            # Zero out gradients
            self.optimizer.zero_grad()

            # Backpropagation
            # actor_critic_loss.backward()
            ppo_loss.backward()

            # Clip gradient norms
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)

            # Step gradients forward
            self.optimizer.step()

        # Step linear annealing process forward
        frac = 1.0 - (episode_num - 1.0) / 100_000
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow

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
                try:
                    state = np.vstack(self.env.reset())
                except:
                    state = np.vstack(self.env.reset()[0])
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

                if self.is_continuous:
                    (action_mus, action_sigmas), V_x = self.actor_critic(torch_state)
                    action_distribution = distributions.Normal(action_mus[0], action_sigmas[0])
                    # action_distribution = distributions.MultivariateNormal(action_mus, torch.diag(action_sigmas[0]))
                    action = action_distribution.sample()
                    log_prob = action_distribution.log_prob(action)
                    entropy = action_distribution.entropy()
                else:
                    # Get action probabilities and V_x from the ActorCritic module and sample action index
                    action_probabilities, V_x = self.actor_critic(torch_state)
                    action_distribution = distributions.Categorical(action_probabilities)
                    action_index = action_distribution.sample()
                    log_prob = action_distribution.log_prob(action_index)
                    entropy = action_distribution.entropy()

                if self.is_gym_env:
                    if not self.is_continuous:
                        action = action_index.item() # Extract numerical value from object

                    # Push the environment one step forward using the sampled action
                    try:
                        state, reward, done, _ = self.env.step(action)
                    except:
                        try:
                            state, reward, done, _, _ = self.env.step(action)
                        except:
                            try:
                                state, reward, done, _ = self.env.step(action.numpy())
                            except:
                                state, reward, done, _, _ = self.env.step(action.numpy())
                    state = np.vstack(state) # Column vector

                    if self.render:
                        self.env.render()
                else:
                    if not self.is_continuous:
                        action = np.array([self.all_actions[:, action_index]])
                    else:
                        action = np.array([[action]])

                    # Push the environment one step forward using the sampled action
                    reward = -self.cost(state, action)[0, 0]
                    state = self.env(state, action)
                    done = step_num == num_steps_per_episode

                state = (state - states.mean()) / (states.std() + epsilon)
                    
                # Add to running episode reward
                episode_reward += reward

                # Track data over time
                try:
                    action_indices.append(action_index)
                except:
                    pass
                actions.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)
                V_xs.append(V_x)
                rewards.append(reward)

                step_num += 1

            states = torch.cat(states)
            try:
                action_indices = torch.cat(action_indices)
            except:
                pass
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
                actions if self.is_continuous else action_indices,
                log_probs,
                entropies,
                advantages,
                returns,
                ppo_steps,
                ppo_clip,
                episode_num
            )
            training_losses.append(actor_critic_loss)

            # Track training rewards
            training_rewards.append(rewards)

            # Compute the mean training reward over the last `num_trials` episodes
            if self.is_gym_env:
                mean_training_rewards = 0
                for episode_rewards in training_rewards[-num_trials:]:
                    mean_training_rewards += sum(episode_rewards)
                mean_training_rewards /= len(training_rewards[-num_trials:])
            else:
                mean_training_rewards = np.mean(np.sum(training_rewards[-num_trials:], axis=1))

            # Log progress every so often
            if episode_num == 1 or episode_num % print_every == 0:
                print(f"| Episode: {episode_num:4} | Episode Reward: {episode_reward:5.1f} | Mean Train Rewards: {mean_training_rewards:5.1f} |")

                # Save models
                if not self.is_gym_env:
                    torch.save(self.actor_critic, f"{self.save_data_path}/{self.actor_critic_file_name}")

                    # Save training data
                    # np.save(f"{self.training_data_path}/{self.controller_type}_v_xs.npy", torch.Tensor(V_xs).detach().numpy())
                    # np.save(f"{self.training_data_path}/{self.controller_type}_v_x_primes.npy", torch.Tensor(V_x_primes).detach().numpy())
                    # np.save(f"{self.training_data_path}/{self.controller_type}_actions.npy", np.array(actions))
                    # np.save(f"{self.training_data_path}/{self.controller_type}_log_probs.npy", torch.Tensor(log_probs).detach().numpy())
                    np.save(f"{self.training_data_path}/{self.controller_type}_training_losses.npy", np.array(training_losses))
                    np.save(f"{self.training_data_path}/{self.controller_type}_rewards.npy", np.array(training_rewards))

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
    reward_threshold = 475 # CartPole-v1
    # reward_threshold = 200 # LunarLandar-v2
    print_every = 10

    ppo = ProximalPolicyOptimization(
        env,
        save_data_path="./analysis/tmp/proximal_policy_optimization",
        learning_rate=0.001,
        gamma=0.99,
        is_gym_env=True,
        is_continuous=False,
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
    total_training_reward_per_episode = []
    for episode_rewards in training_rewards:
        total_training_reward_per_episode.append(sum(episode_rewards))

    training_rewards_slope, training_rewards_intercept = np.polyfit(xs, total_training_reward_per_episode, 1)
    training_rewards_line = training_rewards_slope * xs + training_rewards_intercept

    training_losses_slope, training_losses_intercept = np.polyfit(xs, training_losses, 1)
    training_losses_line = training_losses_slope * xs + training_losses_intercept

    ax = fig.add_subplot(2, 1, 1)
    ax.set_title(f'Training Rewards (Linear Fit Slope = {training_rewards_slope})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(total_training_reward_per_episode, label='Training Reward')
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
