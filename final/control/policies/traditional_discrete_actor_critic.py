import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from final.tensor import KoopmanTensor
from torch.distributions import Categorical, Normal

epsilon = np.finfo(np.float32).eps.item()

class DiscretePolicyIterationPolicy:
    """
        Compute the optimal policy for the given state using discrete Koopman policy iteration methodology.
    """

    def __init__(
        self,
        true_dynamics,
        gamma,
        regularization_lambda,
        koopman_model: KoopmanTensor,
        state_minimums,
        state_maximums,
        all_actions,
        cost,
        save_data_path,
        dt=1.0,
        learning_rate=0.003, # 3e-3
        w_hat_batch_size=2**14,
        seed=123,
        load_model=False,
        layer_1_dim=128,
        layer_2_dim=256
    ):
        """
            Constructor for the DiscretePolicyIterationPolicy class.

            INPUTS:
                true_dynamics - The true dynamics of the system.
                gamma - The discount factor of the system.
                regularization_lambda - The regularization parameter of the policy.
                dynamics_model - The Koopman tensor of the system.
                state_minimums - The minimum values of the state. Should be a column vector.
                state_maximums - The maximum values of the state. Should be a column vector.
                all_actions - The actions that the policy can take. Should be a single dimensional array.
                cost - The cost function of the system. Function must take in states and actions and return scalars.
                save_data_path - The path to save the training data and policy model.
                dt - The time step of the system.
                learning_rate - The learning rate of the policy.
                w_hat_batch_size - The batch size of the policy.
                seed - Random seed for reproducibility.
                load_model - Boolean indicating whether or not to load a saved model.
                layer_1_dim - Dimension of first layer of policy neural network model.
                layer_2_dim - Dimension of second layer of policy neural network model.
        """

        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.policy_file_name = "policy.pt"
        self.value_function_file_name = "value_function.pt"

        self.true_dynamics = true_dynamics
        self.gamma = gamma
        self.regularization_lambda = regularization_lambda
        self.dynamics_model = koopman_model
        self.phi = self.dynamics_model.phi
        self.psi = self.dynamics_model.psi
        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.all_actions = all_actions
        self.cost = cost
        self.save_data_path = save_data_path
        self.training_data_path = f"{self.save_data_path}/training_data"
        self.dt = dt
        self.discount_factor = self.gamma**self.dt
        self.learning_rate = learning_rate
        self.w_hat_batch_size = w_hat_batch_size
        self.layer_1_dim = layer_1_dim
        self.layer_2_dim = layer_2_dim

        if load_model:
            self.policy_model = torch.load(f"{self.save_data_path}/{self.policy_file_name}")
            self.value_function = torch.load(f"{self.save_data_path}/{self.value_function_file_name}")
        else:
            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[1]),
            #     nn.Softmax(dim=-1)
            # )
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, self.layer_1_dim),
                nn.ReLU(),
                # nn.Linear(self.layer_1_dim, self.layer_2_dim),
                # nn.ReLU(),
                nn.Linear(self.layer_1_dim, self.all_actions.shape[1]),
                # nn.Linear(self.layer_2_dim, 2), # Continuous action policy
                nn.Softmax(dim=-1)
            )
            self.value_function = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, self.layer_1_dim),
                nn.ReLU(),
                # nn.Linear(self.layer_1_dim, self.layer_2_dim),
                # nn.ReLU(),
                nn.Linear(self.layer_1_dim, 1)
            )

            # def init_weights(m):
            #     if type(m) == torch.nn.Linear:
            #         m.weight.data.fill_(0.0)

            # self.policy_model.apply(init_weights)
            # self.value_function.apply(init_weights)

        self.policy_model_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.value_function_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=self.learning_rate)

    def get_action(self, state, num_samples=1, is_training=False, epsilon=5):
        # Get action probabilities from policy model
        torch_state = torch.from_numpy(state[:, 0]).float()
        action_probabilities = self.policy_model(torch_state)
        # mu, log_sigma = self.policy_model(torch_state)

        # Create a categorical distribution over the list of probabilities of actions
        policy_distribution = Categorical(action_probabilities)
        # action_distribution = Normal(mu, log_sigma.exp())

        # Choose action uniformly at random if random value is less than epsilon
        if is_training and torch.randint(0, 100, ()) < epsilon:
            uniform_distribution = Categorical(
                torch.ones(self.all_actions.shape[1]) / self.all_actions.shape[1]
            )
            action_indices = uniform_distribution.sample(sample_shape=[num_samples])
        else:
            action_indices = policy_distribution.sample(sample_shape=[num_samples])

        # Compute log probabilities and extract actions from indices
        log_probs = policy_distribution.log_prob(action_indices)
        actions = self.all_actions[:, action_indices]

        return actions, log_probs

    def train(
        self,
        num_training_episodes,
        num_steps_per_episode,
        how_often_to_chkpt=250
    ):
        """
            Actor-Critic algorithm. Train the policy iteration model. This updates the class parameters without returning anything.
            After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.

            INPUTS:
                num_training_episodes - Number of episodes for which to train.
                num_steps_per_episode - Number of steps per episode.
                how_often_to_chkpt - Number of training iterations to do before saving model weights and training data.
        """

        # V_xs = []
        # V_x_primes = []
        # actions = []
        # log_probs = []
        rewards = []

        # Random initial conditions for each episode
        initial_states = np.random.uniform(
            self.state_minimums,
            self.state_maximums,
            [self.dynamics_model.x_dim, num_training_episodes]
        ).T

        for episode_num in range(num_training_episodes):
            # Create arrays to save training data
            V_xs_per_episode = []
            # V_x_primes_per_episode = []
            actions_per_episode = []
            log_probs_per_episode = []
            rewards_per_episode = []

            # Extract initial state
            state = np.vstack(initial_states[episode_num])

            for step_num in range(num_steps_per_episode):
                # Get action and action probability for current state
                action, log_prob = self.get_action(state, is_training=True)
                action = np.array([action])
                # action = action.numpy()
                actions_per_episode.append(action)
                log_probs_per_episode.append(log_prob)

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, action)
                reward = -self.cost(state, action)[0, 0]
                rewards_per_episode.append(reward)

                # Compute V_x and V_x_prime
                V_x = self.value_function(torch.Tensor(state.T))
                V_xs_per_episode.append(V_x)
                # V_x_prime = self.value_function(torch.Tensor(next_state.T))
                # V_x_primes_per_episode.append(V_x_prime)

                # Update state for next loop
                state = next_state

            returns_per_episode = []
            R = 0
            for r in rewards_per_episode[::-1]:
                # Calculate the discounted value
                R = r + self.discount_factor * R
                returns_per_episode.insert(0, R)

            returns_per_episode = torch.tensor(returns_per_episode)
            returns_per_episode = (returns_per_episode - returns_per_episode.mean()) / (returns_per_episode.std() + epsilon)

            policy_losses = []
            value_losses = []
            for R, V_x, log_prob in zip(returns_per_episode, V_xs_per_episode, log_probs_per_episode):
                advantage = R - V_x
                policy_losses.append(-log_prob * advantage.detach())
                value_losses.append(F.mse_loss(V_x, torch.Tensor([[R]])))

            # for reward, V_x_prime, V_x, log_prob in zip(rewards_per_episode, V_x_primes_per_episode, V_xs_per_episode, log_probs_per_episode):
            #     advantage = reward + self.discount_factor*V_x_prime - V_x
            #     policy_losses.append(-log_prob * advantage.detach())
            #     value_losses.append(advantage.pow(2))

            # Reset gradients
            self.policy_model_optimizer.zero_grad()
            self.value_function_optimizer.zero_grad()

            # Compute total loss
            # total_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            total_loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
            # nn.utils.clip_grad_norm_(self.policy_model.parameters(), 100)
            # nn.utils.clip_grad_norm_(self.value_function.parameters(), 100)

            # Backpropagation
            total_loss.backward()
            self.policy_model_optimizer.step()
            self.value_function_optimizer.step()

            # Append episode data arrays to data matrices
            # V_xs.append(V_xs_per_episode)
            # V_x_primes.append(V_x_primes_per_episode)
            # actions.append(actions_per_episode)
            # log_probs.append(log_probs_per_episode)
            rewards.append(rewards_per_episode)

            # Progress prints
            if episode_num == 0 or (episode_num+1) % how_often_to_chkpt == 0:
                total_discounted_reward = 0
                for i in range(num_steps_per_episode-1, -1, -1):
                    total_discounted_reward = rewards_per_episode[i] + (self.discount_factor*total_discounted_reward)
                print(f"Episode: {episode_num+1}, total reward: {sum(rewards_per_episode)} (discounted: {total_discounted_reward})")

                # Save models
                torch.save(self.policy_model, f"{self.save_data_path}/{self.policy_file_name}")
                torch.save(self.value_function, f"{self.save_data_path}/{self.value_function_file_name}")

                # Save training data
                # np.save(f"{self.training_data_path}/v_xs.npy", torch.Tensor(V_xs).detach().numpy())
                # np.save(f"{self.training_data_path}/v_x_primes.npy", torch.Tensor(V_x_primes).detach().numpy())
                # np.save(f"{self.training_data_path}/actions.npy", np.array(actions))
                # np.save(f"{self.training_data_path}/log_probs.npy", torch.Tensor(log_probs).detach().numpy())
                np.save(f"{self.training_data_path}/rewards.npy", np.array(rewards))