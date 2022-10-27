import numpy as np
import torch
import torch.nn as nn

from final.tensor import KoopmanTensor, OLS

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
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[0]),
                nn.Softmax(dim=-1)
            )
            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, self.all_actions.shape[0]),
            #     nn.Softmax(dim=-1)
            # )

            # self.value_function_weights = np.zeros(self.dynamics_model.phi_column_dim)
            self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim)
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim, requires_grad=True)
            # self.value_function_weights = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 1)
            # )

            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)
        
            self.policy_model.apply(init_weights)
            # self.value_function_weights.apply(init_weights)

        self.policy_model_optimizer = torch.optim.Adam(self.policy_model.parameters(), self.learning_rate)
        # self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], self.learning_rate)
        # self.value_function_optimizer = torch.optim.Adam(self.value_function_weights.parameters(), self.learning_rate)

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
        phi_x_batch = self.dynamics_model.phi(x_batch) # (phi_dim, w_hat_batch_size)

        with torch.no_grad():
            pi_response = self.policy_model(torch.Tensor(x_batch.T)).T # (all_actions.shape[0], w_hat_batch_size)

        phi_x_prime_batch = self.dynamics_model.K_(np.array([self.all_actions])) @ phi_x_batch # (all_actions.shape[0], phi_dim, w_hat_batch_size)
        phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response.data.numpy()) # (all_actions.shape[0], phi_dim, w_hat_batch_size)
        expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

        reward_batch_prob = np.einsum(
            'uw,uw->wu',
            -self.cost(x_batch, np.array([self.all_actions])),
            pi_response.data.numpy()
        ) # (w_hat_batch_size, all_actions.shape[0])
        expectation_term_2 = np.array([
            np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
        ]) # (1, w_hat_batch_size)

        self.value_function_weights = torch.linalg.lstsq(
            torch.Tensor((phi_x_batch - (self.discount_factor*expectation_term_1)).T),
            torch.Tensor(expectation_term_2.T)
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
            states = []
            # actions = []
            log_probs_history = []
            critic_value_history = []
            rewards_history = []
            episode_reward = 0

            state = np.vstack(initial_states[episode])
            for step in range(num_steps_per_episode):
                # Add newest state to list of previous states
                states.append(torch.Tensor(state[:,0]))

                # Get action and action probabilities for current state
                action, log_prob = self.get_action(state)
                # actions.append(action)
                log_probs_history.append(log_prob)

                # Compute V_x
                # V_x = torch.Tensor(self.value_function_weights.T @ self.phi(state))
                V_x = self.value_function_weights.T @ torch.Tensor(self.phi(state))
                # V_x = self.value_function_weights(torch.Tensor(state[:,0]))[0]
                critic_value_history.append(V_x)

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, np.array([[action]]))
                curr_reward = -self.cost(state, action)[0,0]
                # rewards[step] = curr_reward

                total_reward_episode[episode] += (self.gamma**(step * self.dt)) * curr_reward

                rewards_history.append(curr_reward)
                episode_reward += curr_reward

                # Update state for next loop
                state = next_state

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = torch.zeros(len(rewards_history))
            discounted_sum = 0
            for i in range(len(rewards_history)-1, -1, -1):
                r = rewards_history[i]
                discounted_sum = r + (self.gamma**self.dt)*discounted_sum
                returns[i] = discounted_sum

            # Normalize returns
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # Calculating loss values to update our networks
            actor_losses = torch.zeros(len(log_probs_history))
            # critic_losses = torch.zeros(len(log_probs_history))
            for i, (log_prob, value, ret) in enumerate(zip(log_probs_history, critic_value_history, returns)):
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up receiving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                # actor_losses.append(-log_prob * diff)  # actor loss
                actor_losses[i] = -log_prob * diff

                # The critic must be updated so that it predicts a better estimate of the future rewards
                # critic_losses.append(torch.pow(ret - value, 2))
                # critic_losses[i] = torch.pow(ret - value, 2)

            # Compute loss
            loss_value = actor_losses.sum() #+ critic_losses.sum()

            # Backpropagation
            self.policy_model_optimizer.zero_grad()
            # self.value_function_optimizer.zero_grad()
            loss_value.backward()
            self.policy_model_optimizer.step()
            # self.value_function_optimizer.step()

            # Update value function weights
            self.update_value_function_weights()

            # Progress prints
            if episode == 0 or (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self.policy_model, self.saved_file_path)
                torch.save(self.value_function_weights, self.saved_file_path_value_function_weights)