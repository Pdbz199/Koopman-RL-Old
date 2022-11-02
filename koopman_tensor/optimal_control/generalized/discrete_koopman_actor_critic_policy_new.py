import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../')
from tensor import KoopmanTensor

class DiscreteKoopmanActorCriticPolicy:
    """
        Compute the optimal policy for the given state using discrete Koopman actor critic methodology.
    """

    def __init__(
        self,
        true_dynamics,
        gamma,
        dynamics_model: KoopmanTensor,
        state_minimums,
        state_maximums,
        all_actions,
        cost,
        saved_file_path,
        dt=1.0,
        learning_rate=0.0003,
        w_hat_batch_size=2**12,
        seed=123,
        load_model=False
    ):
        """
            Constructor for the DiscreteKoopmanPolicyIterationPolicy class.

            INPUTS:
                true_dynamics: The true dynamics of the system.
                gamma: The discount factor of the system.
                dynamics_model: The Koopman tensor of the system.
                state_minimums: The minimum values of the state. Should be a column vector.
                state_maximums: The maximum values of the state. Should be a column vector.
                all_actions: The actions that the policy can take. Should be a single dimensional array.
                cost: The cost function of the system. Function must take in states and actions and return scalars.
                saved_file_path: The path to save the policy model.
                dt: The time step of the system.
                learning_rate: The learning rate of the policy.
                w_hat_batch_size: The batch size of the policy.
        """

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.true_dynamics = true_dynamics
        self.gamma = gamma
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
            self.saved_file_path_value_function_weights = split_path[0] + '-value_function_weights.pt'
        else:
            self.saved_file_path_value_function_weights = split_path[0] + '-value_function_weights.' + split_path[1]
        self.dt = dt
        self.learning_rate = learning_rate
        self.w_hat_batch_size = w_hat_batch_size

        if load_model:
            self.policy_model = torch.load(self.saved_file_path) # actor model
            self.value_function_weights = torch.load(self.saved_file_path_value_function_weights) # critic weights
        else:
            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[0]),
            #     nn.Softmax(dim=-1)
            # ) # actor model

            # self.policy_model = nn.Sequential(
            #     nn.Linear(self.dynamics_model.phi_dim, self.all_actions.shape[0]),
            #     nn.Softmax(dim=-1)
            # ) # actor model

            layer_1_dim = 512
            layer_2_dim = 256
            # layer_3_dim = 128
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, layer_1_dim),
                nn.ReLU(),
                nn.Linear(layer_1_dim, layer_2_dim),
                nn.ReLU(),
                nn.Linear(layer_2_dim, self.all_actions.shape[0]),
                # nn.ReLU(),
                # nn.Linear(layer_3_dim, self.all_actions.shape[0]),
                nn.Softmax(dim=-1)
            )

            self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim)
            # self.value_function_weights = torch.zeros(self.dynamics_model.phi_column_dim, requires_grad=True)

            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    m.weight.data.fill_(0.0)

            self.policy_model.apply(init_weights)

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        # self.critic_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)

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
        rewards = np.zeros([self.all_actions.shape[1], self.w_hat_batch_size])
        for i in range(self.w_hat_batch_size):
            for j in range(self.all_actions.shape[1]):
                rewards[j][i] = -self.cost(np.vstack(x_batch[:,i]), np.vstack(self.all_actions[:,j]))
        rewards = torch.Tensor(rewards)
        reward_batch_prob = (rewards * pi_response).T # (w_hat_batch_size, all_actions.shape[1])
        reward_batch_prob_sum = torch.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
        expectation_term_2 = reward_batch_prob_sum.reshape([1,reward_batch_prob_sum.shape[0]]) # (1, w_hat_batch_size)

        # Compute optimized value function weights using OLS
        self.value_function_weights = torch.linalg.lstsq(
            (torch.Tensor(phi_x_batch) - ((self.gamma**self.dt)*expectation_term_1)).T,
            torch.Tensor(expectation_term_2.T)
        ).solution

    def get_action(self, x, num_samples=1):
        """
            Estimate the policy distribution and sample an action, compute its log probability.
            
            INPUTS:
                x - State column vector.
            OUTPUTS:
                The selected action(x) and log probability(ies).
        """

        action_probabilities = self.policy_model(torch.Tensor(x[:,0]))
        action_indices = torch.multinomial(action_probabilities, num_samples).item()
        actions = torch.Tensor(np.vstack(self.all_actions[:,action_indices]))
        log_probs = torch.log(action_probabilities[action_indices])

        return actions, log_probs

    # def get_action(self, s):
    #     """
    #         INPUTS:
    #             s - 1D state array    
    #     """
        
    #     action_probs = self.policy_model(torch.Tensor(s))
    #     action_index = np.random.choice(self.all_actions.shape[0], p=np.squeeze(action_probs.detach().numpy()))
    #     action = self.all_actions[action_index]
    #     return action

    def actor_critic(
        self,
        num_training_episodes,
        num_steps_per_episode
    ):
        """
            REINFORCE algorithm
                
            INPUTS:
                num_training_episodes: number of episodes to train for
                num_steps_per_episode: number of steps per episode
        """

        # Initialize R_bar (Average reward)
        # R_bar = 0.0
        epsilon = np.finfo(np.float32).eps.item()

        # Initialize S
        initial_states = np.random.uniform(
            self.state_minimums,
            self.state_maximums,
            [self.dynamics_model.x_dim, num_training_episodes]
        )
        total_reward_episode = torch.zeros(num_training_episodes)

        for episode in range(num_training_episodes):
            # log_probs = []
            # V_xs = []
            # rewards = []

            log_probs = torch.zeros(num_steps_per_episode)
            V_xs = torch.zeros(num_steps_per_episode)
            rewards = torch.zeros(num_steps_per_episode)

            state = np.vstack(initial_states[:,episode])
            for step in range(num_steps_per_episode):
                # Get action and log probability for current state
                action, log_prob = self.get_action(state)
                # log_probs.append(log_prob)
                log_probs[step] = log_prob

                # Compute V_x
                # V_x = torch.Tensor(self.value_function_weights.T @ self.phi(np.vstack(state)))
                V_x = self.value_function_weights.T @ torch.Tensor(self.phi(state))
                # V_xs.append(V_x)
                V_xs[step] = V_x

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, action.numpy())
                curr_reward = -self.cost(state, action.numpy())[0,0]

                total_reward_episode[episode] += curr_reward
                # total_reward_episode[episode] += (self.gamma**(step*self.dt)) * curr_reward
                # total_reward_episode[episode] += self.gamma**step * curr_reward

                # rewards.append(curr_reward)
                rewards[step] = curr_reward

                # Update state for next loop
                state = next_state
            
            # log_probs = torch.Tensor(log_probs)
            # V_xs = torch.Tensor(V_xs)
            # rewards = torch.Tensor(rewards)

            # Calculate true V(x) from rewards
            # returns = []
            returns = torch.zeros(num_steps_per_episode)
            R = 0
            for i in range(len(rewards)-1, -1, -1):
                R = rewards[i] + self.gamma*R
                # returns.insert(0, R)
                returns[i] = R
            # returns = torch.Tensor(returns)

            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + epsilon)

            # Calculating loss to update our network(s)
            advantage = returns - V_xs
            actor_loss = (-log_probs * advantage).mean()
            # critic_loss = advantage.pow(2).mean()
            loss = actor_loss #+ critic_loss

            # Update policy model with backpropagation
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            # Update value function weights
            self.update_value_function_weights()

            if episode == 0 or (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self.policy_model, self.saved_file_path)
                torch.save(self.value_function_weights, self.saved_file_path_value_function_weights)