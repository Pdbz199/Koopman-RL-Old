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
            self.value_function_weights = torch.load(self.saved_file_path_value_function_weights).numpy()
        else:
            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[1]),
            #     nn.Softmax(dim=-1)
            # )
            layer_1_dim = 512
            layer_2_dim = 256
            # layer_3_dim = 128
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, layer_1_dim),
                nn.ReLU(),
                nn.Linear(layer_1_dim, layer_2_dim),
                nn.ReLU(),
                nn.Linear(layer_2_dim, self.all_actions.shape[1]),
                # nn.ReLU(),
                # nn.Linear(layer3_dim, self.all_actions.shape[1]),
                nn.Softmax(dim=-1)
            )
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim)
            self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim, requires_grad=True)

            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)
        
            self.policy_model.apply(init_weights)

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)

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
        actions = torch.Tensor(np.vstack(self.all_actions[:,action_indices]))
        log_probs = torch.log(action_probabilities[action_indices])

        return actions, log_probs

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
        K_us = self.dynamics_model.K_(self.all_actions)
        K_us = torch.Tensor(K_us)
        phi_x_prime_batch = torch.zeros([self.all_actions.shape[1], self.dynamics_model.phi_dim, self.w_hat_batch_size])
        for i in range(K_us.shape[0]):
            phi_x_prime_batch[i] = K_us[i] @ phi_x_batch
        
        # Expected phi(x')
        phi_x_prime_batch_prob = torch.einsum('upw,uw->upw', phi_x_prime_batch, pi_response) # (all_actions.shape[1], phi_dim, w_hat_batch_size)
        expectation_term_1 = torch.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

        # Expected reward
        # rewards = torch.Tensor(-self.cost(x_batch, self.all_actions)) # (all_actions.shape[1], w_hat_batch_size)
        rewards = np.zeros([self.all_actions.shape[1], self.w_hat_batch_size])
        for i in range(self.w_hat_batch_size):
            for j in range(self.all_actions.shape[1]):
                rewards[j][i] = -self.cost(np.vstack(x_batch[:,i]), np.array([self.all_actions[:,j]]))
        rewards = torch.Tensor(rewards)
        reward_batch_prob = (rewards * pi_response).T # (w_hat_batch_size, all_actions.shape[1])
        reward_batch_prob_sum = torch.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
        expectation_term_2 = reward_batch_prob_sum.reshape([1,reward_batch_prob_sum.shape[0]]) # (1, w_hat_batch_size)

        # Compute w hat using OLS
        self.value_function_weights = torch.linalg.lstsq(
            (torch.Tensor(phi_x_batch) - (self.discount_factor*expectation_term_1)).T,
            torch.Tensor(expectation_term_2.T)
        ).solution

    def update_policy_model(self, returns, log_probs):
        """
            Update the weights of the policy network given the training samples.
            
            INPUTS:
                returns - Return (cumulative rewards) for each step in an episode.
                log_probs - Log probability for each step.
        """

        policy_gradient = torch.zeros(log_probs.shape[0])
        for i, (log_prob, R_t) in enumerate(zip(log_probs, returns)):
            policy_gradient[i] = -log_prob * self.gamma**((len(returns)-i) * self.dt) * R_t

        loss = policy_gradient.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Q(self, x, u):
        """
            Q function using latest value function weights.

            INPUTS:
                x - State column vector.
                u - Action column vector.
            
            OUTPUTS:
                Quality value for the given state action pair.
        """
        
        V_x_prime = self.value_function_weights.T @ self.dynamics_model.phi_f(x, u)
        
        return (torch.Tensor(-self.cost(x, u)) + self.discount_factor*V_x_prime)[0,0]

    def train(self, num_training_episodes, num_steps_per_episode):
        """
            REINFORCE algorithm. Train the policy iteration model. This updates the class parameters without returning anything.
            After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.
                
            INPUTS:
                num_training_episodes - Number of episodes for which to train.
                num_steps_per_episode - Number of steps per episode.
        """

        epsilon = torch.finfo(torch.float64).eps
        total_reward_episode = torch.zeros(num_training_episodes)

        initial_states = np.random.uniform(
            self.state_maximums,
            self.state_minimums,
            [self.dynamics_model.x_dim, num_training_episodes]
        ).T

        for episode in range(num_training_episodes):
            states = torch.zeros([num_steps_per_episode, self.dynamics_model.x_dim])
            actions = torch.zeros(num_steps_per_episode)
            log_probs = torch.zeros(num_steps_per_episode)
            rewards = torch.zeros(num_steps_per_episode)

            state = np.vstack(initial_states[episode])

            for step in range(num_steps_per_episode):
                states[step] = torch.Tensor(state[:,0])

                action, log_prob = self.get_action(state)
                actions[step] = action
                log_probs[step] = log_prob

                curr_reward = -self.cost(state, action.numpy())[0,0]
                rewards[step] = curr_reward

                total_reward_episode[episode] += self.gamma**(step*self.dt) * curr_reward

                state = self.true_dynamics(state, action.numpy())

            # Estimate returns using Q function with latest value function weights
            true_returns = torch.zeros(num_steps_per_episode)
            estimated_returns = torch.zeros(num_steps_per_episode)
            R = 0
            for i in range(num_steps_per_episode-1, -1, -1):
                R = rewards[i] + self.discount_factor*R
                true_returns[i] = R

                Q_val = self.Q(
                    np.vstack(states[i]),
                    np.vstack([[actions[i]]])
                )
                estimated_returns[i] = Q_val

            true_returns = (true_returns - true_returns.mean()) / (true_returns.std() + epsilon)
            estimated_returns = (estimated_returns - estimated_returns.mean()) / (estimated_returns.std() + epsilon)

            # Update policy model and value function weights
            self.update_policy_model(estimated_returns, log_probs)
            value_function_loss = (true_returns - estimated_returns).pow(2).mean()
            self.value_function_optimizer.zero_grad()
            value_function_loss.backward()
            self.value_function_optimizer.step()
            # self.update_value_function_weights()

            # Progress prints
            if episode == 0 or (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self.policy_model, self.saved_file_path)
                torch.save(torch.Tensor(self.value_function_weights), self.saved_file_path_value_function_weights)