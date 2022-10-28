import numpy as np
import torch
import torch.nn as nn

from final.tensor import KoopmanTensor

class DiscreteKoopmanPolicyIterationPolicy:
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
        saved_file_path,
        dt=1.0,
        learning_rate=0.003,
        w_hat_batch_size=2**12,
        seed=123,
        load_model=False
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
                saved_file_path - The path to save the policy model.
                dt - The time step of the system.
                learning_rate - The learning rate of the policy.
                w_hat_batch_size - The batch size of the policy.
                seed - Random seed for reproducibility.
                load_model - Boolean indicating whether or not to load a saved model.
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
        self.saved_file_path = saved_file_path
        split_path = self.saved_file_path.split('.')
        if len(split_path) == 1:
            self.saved_file_path_value_function_weights = split_path[0] + '-value-function-weights.pt'
        else:
            self.saved_file_path_value_function_weights = split_path[0] + '-value-function-weights.' + split_path[1]
        self.dt = dt
        self.discount_factor = self.gamma**self.dt
        self.learning_rate = learning_rate
        self.w_hat_batch_size = w_hat_batch_size

        if load_model:
            self.policy_model = torch.load(self.saved_file_path)
            self.value_function_weights = torch.load(self.saved_file_path_value_function_weights)#.numpy()
        else:
            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[0]),
            #     nn.Softmax(dim=-1)
            # )
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, self.all_actions.shape[0]),
                nn.Softmax(dim=-1)
            )

            # self.value_function_weights = np.zeros(self.dynamics_model.phi_column_dim)
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim)
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim, requires_grad=True)
            self.value_function_weights = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)
        
            self.policy_model.apply(init_weights)
            self.value_function_weights.apply(init_weights)

        self.policy_model_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        # self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)
        self.value_function_optimizer = torch.optim.Adam(self.value_function_weights.parameters(), lr=self.learning_rate)

    def get_action(self, x, num_samples=1):
        """
            Estimate the policy distribution and sample an action, compute its log probability.
            
            INPUTS:
                s - State column vector.
            OUTPUTS:
                The selected action(s) and log probability(ies).
        """

        action_probabilities = self.policy_model(torch.Tensor(x[:,0]))
        action_indices = torch.multinomial(action_probabilities, num_samples).item()
        actions = self.all_actions[action_indices].item()
        log_probs = torch.log(action_probabilities[action_indices])

        return actions, log_probs

    def update_value_function_weights(self):
        """
            Update the weights for the value function in the dictionary space.
        """

        x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], self.w_hat_batch_size, replace=False)
        x_batch = self.dynamics_model.X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
        phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices] # (phi_dim, w_hat_batch_size)

        with torch.no_grad():
            pi_response = self.policy_model(torch.Tensor(x_batch.T)).T # (all_actions.shape[0], w_hat_batch_size)

        K_us = self.dynamics_model.K_(np.array([self.all_actions])) # (all_actions.shape[0], phi_dim, phi_dim)
        K_us = torch.Tensor(K_us)
        phi_x_prime_batch = torch.zeros([self.all_actions.shape[0], self.dynamics_model.phi_dim, self.w_hat_batch_size])
        for i in range(K_us.shape[0]):
            phi_x_prime_batch[i] = K_us[i] @ phi_x_batch
        phi_x_prime_batch_prob = torch.einsum(
            'upw,uw->upw',
            torch.Tensor(phi_x_prime_batch),
            pi_response
        ) # (all_actions.shape[0], phi_dim, w_hat_batch_size)
        expectation_term_1 = torch.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

        reward_batch_prob = (torch.Tensor(-self.cost(x_batch, np.array([self.all_actions]))) * pi_response).T
        reward_batch_prob_sum = torch.sum(reward_batch_prob, axis=1)
        expectation_term_2 = reward_batch_prob_sum.reshape([1,reward_batch_prob_sum.shape[0]])

        self.value_function_weights = torch.linalg.lstsq(
            (torch.Tensor(phi_x_batch) - (self.discount_factor*expectation_term_1)).T,
            expectation_term_2.T
        ).solution

    def train(self, num_training_episodes, num_steps_per_episode):
        """
            actor-critic algorithm. Train the policy iteration model. This updates the class parameters without returning anything.
            After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.
                
            INPUTS:
                num_training_episodes - Number of episodes for which to train.
                num_steps_per_episode - Number of steps per episode.
        """

        initial_states = np.random.uniform(
            self.state_minimums,
            self.state_maximums,
            [self.dynamics_model.x_dim, num_training_episodes]
        ).T
        total_reward_episode = torch.zeros(num_training_episodes)

        for episode in range(num_training_episodes):
            critic_values = torch.zeros(num_steps_per_episode)
            log_probs = torch.zeros(num_steps_per_episode)
            rewards = torch.zeros(num_steps_per_episode)
            episode_reward = 0

            state = np.vstack(initial_states[episode])
            for step in range(num_steps_per_episode):
                # Compute V_x
                # V_x = torch.Tensor(self.value_function_weights.T @ self.phi(state))
                # V_x = self.value_function_weights.T @ torch.Tensor(self.phi(state))
                V_x = self.value_function_weights(torch.Tensor(state[:,0]))[0]
                critic_values[step] = V_x

                # Get action and action probabilities for current state
                action, log_prob = self.get_action(state)
                log_probs[step] = log_prob

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, np.array([[action]]))
                curr_reward = -self.cost(state, action)[0,0]
                rewards[step] = curr_reward
                episode_reward += curr_reward

                total_reward_episode[episode] += (self.gamma**(step * self.dt)) * curr_reward

                # Update state for next loop
                state = next_state

            # Calculate expected value from rewards
            returns = torch.zeros(num_steps_per_episode)
            discounted_sum = 0
            for i in range(num_steps_per_episode-1, -1, -1):
                r = rewards[i]
                discounted_sum = r + (self.gamma**self.dt)*discounted_sum
                returns[i] = discounted_sum

            # Normalize returns
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # Detach returns from gradient graph
            returns = returns.detach()

            # Compute advantage
            advantage = returns - critic_values

            # Compute actor and critic losses
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            # Backpropagation
            self.policy_model_optimizer.zero_grad()
            self.value_function_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.policy_model_optimizer.step()
            self.value_function_optimizer.step()

            # Update value function weights
            # self.update_value_function_weights()

            # Progress prints
            if episode == 0 or (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self.policy_model, self.saved_file_path)
                torch.save(self.value_function_weights, self.saved_file_path_value_function_weights)