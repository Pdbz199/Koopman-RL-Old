import numpy as np
import torch
import torch.nn as nn

from final.tensor import KoopmanTensor
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

class ContinuousKoopmanPolicyIterationPolicy:
    """
        Compute the optimal policy for the given state using discrete Koopman policy iteration methodology.
    """

    def __init__(
        self,
        true_dynamics,
        gamma,
        regularization_lambda,
        dynamics_model: KoopmanTensor,
        state_minimums,
        state_maximums,
        all_actions,
        cost,
        save_data_path,
        dt=1.0,
        learning_rate=0.003,
        w_hat_batch_size=2**12,
        seed=123,
        load_model=False,
        layer_1_dim=256,
        layer_2_dim=128
    ):
        """
            Constructor for the DiscreteKoopmanPolicyIterationPolicy class.

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

        self.true_dynamics = true_dynamics
        self.gamma = gamma
        self.regularization_lambda = regularization_lambda
        self.dynamics_model = dynamics_model
        self.phi = self.dynamics_model.phi
        self.psi = self.dynamics_model.psi
        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.all_actions = all_actions
        self.cost = cost
        self.save_data_path = save_data_path
        self.value_function_weights_file_name = "value_function_weights.npy"
        # self.value_function_weights_file_name = "value_function_weights.pt"
        self.dt = dt
        self.discount_factor = self.gamma**self.dt
        self.learning_rate = learning_rate
        self.w_hat_batch_size = w_hat_batch_size
        self.layer_1_dim = layer_1_dim
        self.layer_2_dim = layer_2_dim

        if load_model:
            self.mu_model = torch.load(f"{self.save_data_path}/mu_model.pt")
            self.log_sigma_model = torch.load(f"{self.save_data_path}/log_sigma_model.pt")
            self.value_function_weights = np.load(f"{self.save_data_path}/{self.value_function_weights_file_name}")
        else:
            # self.mu_model = nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[0])
            self.mu_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, self.layer_1_dim),
                nn.ReLU(),
                nn.Linear(self.layer_1_dim, self.layer_2_dim),
                nn.ReLU(),
                nn.Linear(self.layer_2_dim, self.all_actions.shape[0]),
                nn.Softmax(dim=-1)
            )
            self.log_sigma_model = torch.zeros(self.all_actions.shape[0], requires_grad=True)

            self.value_function_weights = np.zeros(self.dynamics_model.phi_column_dim)

            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)
        
            self.mu_model.apply(init_weights)
            # self.value_function_weights.apply(init_weights)

        self.policy_model_optimizer = torch.optim.Adam(list(self.mu_model.parameters()) + [self.log_sigma_model], lr=self.learning_rate)
        # self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)

    def get_action_distribution(self, x):
        """
            Get the action distribution for a given state.

            INPUTS:
                x - State column vector.

            OUTPUTS:
                Normal distribution using state dependent mu and latest sigma.
        """

        mu = self.mu_model(torch.Tensor(x[:, 0]))
        sigma = torch.exp(self.log_sigma_model)

        # return MultivariateNormal(mu, torch.diag(sigma))
        return Normal(mu, sigma)

    def get_action(self, x):
        """
            Estimate the policy and sample an action, compute its log probability.
            
            INPUTS:
                x - State column vector.
            
            OUTPUTS:
                The selected action and log probability.
        """

        action_distribution = self.get_action_distribution(x)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action, log_prob

    def update_value_function_weights(self):
        """
            Update the weights for the value function in the dictionary space.
        """

        # Take state sample from dataset
        x_batch_indices = np.random.choice(
            self.dynamics_model.X.shape[1],
            self.w_hat_batch_size,
            replace=False
        )
        x_batch = self.dynamics_model.X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
        phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices] # (phi_dim, w_hat_batch_size)

        # Compute policy probabilities for each state in the batch
        with torch.no_grad():
            pi_response = np.zeros([self.all_actions.shape[1],self.w_hat_batch_size])
            for state_index, state in enumerate(x_batch.T):
                action_distribution = self.get_action_distribution(np.vstack(state))
                pi_response[:, state_index] = action_distribution.log_prob(torch.Tensor(self.all_actions.T))[:, 0]

        # Compute phi_x_prime for all states in the batch using all actions in the action space
        K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
        phi_x_prime_batch = K_us @ phi_x_batch # (all_actions.shape[1], phi_dim, w_hat_batch_size)

        # Compute expected phi(x')
        phi_x_prime_batch_prob = np.einsum(
            'upw,uw->upw',
            phi_x_prime_batch,
            pi_response
        ) # (all_actions.shape[1], phi_dim, w_hat_batch_size)
        expectation_term_1 = phi_x_prime_batch_prob.sum(axis=0) # (phi_dim, w_hat_batch_size)

        # Compute expected reward
        rewards_batch = -self.cost(x_batch, self.all_actions) # (all_actions.shape[1], w_hat_batch_size)
        reward_batch_prob = np.einsum(
            'uw,uw->wu',
            rewards_batch,
            pi_response
        ) # (w_hat_batch_size, all_actions.shape[1])
        expectation_term_2 = np.array([
            reward_batch_prob.sum(axis=1) # (w_hat_batch_size,)
        ]) # (1, w_hat_batch_size)

        # Update value function weights
        self.value_function_weights = torch.linalg.lstsq(
            torch.Tensor((phi_x_batch - (self.discount_factor * expectation_term_1)).T),
            torch.Tensor(expectation_term_2.T)
        ).solution.numpy() # (phi_dim, 1)

    def train(self, num_training_episodes, num_steps_per_episode):
        """
            Actor-Critic algorithm. Train the policy iteration model. This updates the class parameters without returning anything.
            After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.

            INPUTS:
                num_training_episodes - Number of episodes for which to train.
                num_steps_per_episode - Number of steps per episode.
        """

        # epsilon = np.finfo(np.float32).eps.item()
        V_xs = []
        V_x_primes = []
        actions = []
        log_probs = []
        rewards = []

        initial_states = np.random.uniform(
            self.state_minimums,
            self.state_maximums,
            [self.dynamics_model.x_dim, num_training_episodes]
        ).T

        for episode_num in range(num_training_episodes):
            # Create arrays to save training data
            V_xs_per_episode = torch.zeros(num_steps_per_episode)
            V_x_primes_per_episode = torch.zeros_like(V_xs_per_episode)
            actions_per_episode = torch.zeros((num_steps_per_episode, self.all_actions.shape[1]))
            log_probs_per_episode = torch.zeros_like(V_xs_per_episode)
            rewards_per_episode = torch.zeros_like(V_xs_per_episode)

            # Extract initial state
            state = np.vstack(initial_states[episode_num])

            for step_num in range(num_steps_per_episode):
                # Compute V_x
                V_x = self.value_function_weights.T @ self.phi(state)
                V_xs_per_episode[step_num] = V_x[0, 0] # ()

                # Get action and action probabilities for current state
                action, log_prob = self.get_action(state)
                action = action.reshape(self.all_actions.shape[0], 1)
                numpy_action = action.detach().numpy()
                actions_per_episode[step_num] = action[:, 0] # (action dim,)
                log_probs_per_episode[step_num] = log_prob[0] # ()

                # Compute V_x_prime
                # V_x_prime = self.value_function_weights.T @ self.dynamics_model.phi_f(state, action)
                V_x_prime = self.value_function_weights.T @ self.dynamics_model.phi(self.dynamics_model.f(state, numpy_action))
                V_x_primes_per_episode[step_num] = V_x_prime[0, 0] # ()

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, numpy_action)
                earned_reward = -self.cost(state, numpy_action)[0, 0]
                rewards_per_episode[step_num] = earned_reward # ()

                # Update state for next loop
                state = next_state

            # Append data to arrays
            V_xs.append(V_xs_per_episode.detach().numpy())
            V_x_primes.append(V_x_primes_per_episode.detach().numpy())
            actions.append(actions_per_episode.detach().numpy())
            log_probs.append(log_probs_per_episode.detach().numpy())
            rewards.append(rewards_per_episode.detach().numpy())

            # Compute advantage
            target_V_xs_per_episode = rewards_per_episode + (self.discount_factor * V_x_primes_per_episode)# - \
                #self.regularization_lambda * log_probs_per_episode
            advantage_per_episode = target_V_xs_per_episode - V_xs_per_episode

            # Compute actor and critic losses
            actor_loss = (-log_probs_per_episode * advantage_per_episode.detach()).mean()
            # actor_loss = (self.regularization_lambda * log_probs_per_episode - V_xs_per_episode).mean()

            # Backpropagation
            self.policy_model_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_model_optimizer.step()

            # Update value function weights
            self.update_value_function_weights()

            # Progress prints
            if episode_num == 0 or (episode_num+1) % 250 == 0:
                # print(f"Episode: {episode_num+1}, total discounted reward: {total_reward_per_episode[episode_num]}")
                print(f"Episode: {episode_num+1}, total reward: {rewards_per_episode.sum()}")

                # Save models
                torch.save(self.mu_model, f"{self.save_data_path}/mu_model.pt")
                torch.save(self.log_sigma_model, f"{self.save_data_path}/log_sigma_model.pt")
                torch.save(self.value_function_weights, f"{self.save_data_path}/{self.value_function_weights_file_name}")

                # Save training data
                training_data_path = f"{self.save_data_path}/training_data"
                np.save(f"{training_data_path}/v_xs.npy", np.array(V_xs))
                np.save(f"{training_data_path}/v_x_primes.npy", np.array(V_x_primes))
                np.save(f"{training_data_path}/actions.npy", np.array(actions))
                np.save(f"{training_data_path}/log_probs.npy", np.array(log_probs))
                np.save(f"{training_data_path}/rewards.npy", np.array(rewards))
                # np.save(f"{training_data_path}/total_reward_per_episode.npy", total_reward_per_episode.numpy())