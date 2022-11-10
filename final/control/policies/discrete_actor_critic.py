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
                saved_file_path - The path to save the policy model.
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
        self.layer_1_dim = layer_1_dim
        self.layer_2_dim = layer_2_dim

        if load_model:
            self.policy_model = torch.load(self.saved_file_path)
            self.value_function_weights = torch.load(self.saved_file_path_value_function_weights).numpy()
        else:
            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[1]),
            #     nn.Softmax(dim=-1)
            # )
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, self.layer_1_dim),
                nn.ReLU(),
                nn.Linear(self.layer_1_dim, self.layer_2_dim),
                nn.ReLU(),
                nn.Linear(self.layer_2_dim, self.all_actions.shape[1]),
                nn.Softmax(dim=-1)
            )

            # self.value_function_weights = np.zeros(self.dynamics_model.phi_column_dim)
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim)
            self.value_function_weights = np.zeros(self.dynamics_model.phi_column_dim)
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim, requires_grad=True)
            # self.value_function_weights = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 1)
            # )

            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)
        
            self.policy_model.apply(init_weights)
            # self.value_function_weights.apply(init_weights)

        self.policy_model_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        # self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)
        # self.value_function_optimizer = torch.optim.Adam(self.value_function_weights.parameters(), lr=self.learning_rate)

    def update_value_function_weights(self):
        """
            Update the weights for the value function in the dictionary space.
        """

        # Take state sample from dataset
        x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], self.w_hat_batch_size, replace=False)
        x_batch = self.dynamics_model.X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
        phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices] # (phi_dim, w_hat_batch_size)

        # Compute policy probabilities for each state in the batch
        with torch.no_grad():
            pi_response = self.policy_model(torch.Tensor(x_batch.T)).T # (all_actions.shape[1], w_hat_batch_size)

        # Compute phi_x_prime for all states in the batch using all actions in the action space
        phi_x_prime_batch = self.dynamics_model.K_(self.all_actions) @ phi_x_batch # (all_actions.shape[1], phi_dim, w_hat_batch_size)

        # Compute expected phi(x')
        phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response.numpy()) # (all_actions.shape[1], phi_dim, w_hat_batch_size)
        expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

        # Compute expected reward
        rewards_batch = -self.cost(x_batch, self.all_actions) # (all_actions.shape[1], w_hat_batch_size)
        reward_batch_prob = np.einsum(
            'uw,uw->wu',
            rewards_batch,
            pi_response.numpy()
        ) # (w_hat_batch_size, all_actions.shape[1])
        expectation_term_2 = np.array([
            np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
        ]) # (1, w_hat_batch_size)

        # Update value function weights
        self.value_function_weights = torch.linalg.lstsq(
            torch.Tensor((phi_x_batch - ((self.gamma**self.dt)*expectation_term_1)).T),
            torch.Tensor(expectation_term_2.T)
        ).solution.numpy() # (phi_dim, 1)

    def get_action(self, x, num_samples=1):
        """
            Estimate the policy distribution, sample actions, and compute the log probabilities of sampled actions.
            
            INPUTS:
                x - State column vector.
                
            OUTPUTS:
                The selected actions and log probabilities.
        """

        # Get action distribution from policy model
        action_probabilities = self.policy_model(torch.Tensor(x[:,0]))

        # Get random index of action from distribution
        action_indices = np.random.choice(
            self.all_actions.shape[1],
            p=np.squeeze(action_probabilities.detach().numpy()),
            size=[num_samples]
        )

        # Select action and compute log probability for that action
        actions = self.all_actions[:,action_indices]
        log_probabilities = torch.log(action_probabilities[action_indices])

        return actions, log_probabilities

    def train(self, num_training_episodes, num_steps_per_episode):
        """
            actor-critic algorithm. Train the policy iteration model. This updates the class parameters without returning anything.
            After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.
                
            INPUTS:
                num_training_episodes - Number of episodes for which to train.
                num_steps_per_episode - Number of steps per episode.
        """

        # epsilon = np.finfo(np.float32).eps.item()
        total_reward_episode = torch.zeros(num_training_episodes)

        initial_states = np.random.uniform(
            self.state_minimums,
            self.state_maximums,
            [self.dynamics_model.x_dim, num_training_episodes]
        ).T

        for episode in range(num_training_episodes):
            V_xs = torch.zeros(num_steps_per_episode)
            V_x_primes = torch.zeros(num_steps_per_episode)
            log_probs = torch.zeros(num_steps_per_episode)
            rewards = torch.zeros(num_steps_per_episode)

            state = np.vstack(initial_states[episode])

            for step in range(num_steps_per_episode):
                # Compute V_x
                # V_x = torch.Tensor(self.value_function_weights.T @ self.phi(state))
                V_x = self.value_function_weights.T @ self.phi(state)
                # V_x = self.value_function_weights(torch.Tensor(state[:,0]))[0]
                # critic_values[step] = V_x
                V_xs[step] = torch.Tensor(V_x)

                # Get action and action probabilities for current state
                action, log_prob = self.get_action(state)
                log_probs[step] = log_prob

                # Compute V_x_prime
                V_x_prime = self.value_function_weights.T @ self.dynamics_model.phi_f(state, action)
                V_x_primes[step] = torch.Tensor(V_x_prime)

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, action)
                curr_reward = -self.cost(state, action)[0,0]
                rewards[step] = curr_reward

                # Add to total discounted reward for the current episode
                total_reward_episode[episode] += (self.gamma**(step*self.dt)) * curr_reward

                # Update state for next loop
                state = next_state

            # Calculate returns for each step in trajectory
            # returns = torch.zeros(num_steps_per_episode)
            # R = 0
            # for i in range(num_steps_per_episode-1, -1, -1):
            #     R = rewards[i] + (self.gamma**self.dt)*R
            #     returns[i] = R

            # Normalize returns
            # returns = (returns - returns.mean()) / (returns.std() + epsilon)

            # Calculating loss values to update our network(s)
            # advantage = returns - V_xs
            # actor_loss = -log_probabilities * advantage
            # critic_loss = torch.pow(advantage, 2)

            # Compute loss
            # loss = actor_loss.sum()
            # loss = torch.sum(actor_loss) + torch.sum(critic_loss)

            # Backpropagation
            # self.policy_optimizer.zero_grad()
            # loss.backward()
            # self.policy_optimizer.step()

            # Update value function weights
            # self.update_value_function_weights()

            # Detach returns from gradient graph
            # returns = returns.detach()

            # Compute advantage
            # advantage = returns - critic_values
            advantage = rewards + V_x_primes - V_xs

            # Compute actor and critic losses
            actor_loss = (-log_probs * advantage.detach()).mean()
            # critic_loss = advantage.pow(2).mean()

            # Backpropagation
            self.policy_model_optimizer.zero_grad()
            # self.value_function_optimizer.zero_grad()
            actor_loss.backward()
            # critic_loss.backward()
            self.policy_model_optimizer.step()
            # self.value_function_optimizer.step()

            # Update value function weights
            self.update_value_function_weights()

            # Progress prints
            if episode == 0 or (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self.policy_model, self.saved_file_path)
                torch.save(self.value_function_weights, self.saved_file_path_value_function_weights)