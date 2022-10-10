import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS

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
        learning_rate=0.003,
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
            self.saved_file_path_w_hat = split_path[0] + '-w_hat.pt'
        else:
            self.saved_file_path_w_hat = split_path[0] + '-w_hat.' + split_path[1]
        self.dt = dt
        self.learning_rate = learning_rate
        self.w_hat_batch_size = w_hat_batch_size

        if load_model:
            self.policy_model = torch.load(self.saved_file_path) # actor model
            self.w_hat = torch.load(self.saved_file_path_w_hat).numpy()  # critic weights
        else:
            self.policy_model = nn.Sequential(
                nn.Linear(self.dynamics_model.x_dim, self.all_actions.shape[0]),
                nn.Softmax(dim=-1)
            ) # actor model
            self.w_hat = np.zeros(self.dynamics_model.phi_column_dim) # critic weights

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), self.learning_rate)
        self.w_hat = np.zeros(self.dynamics_model.phi_column_dim)
        
        def init_weights(m):
            if type(m) == torch.nn.Linear:
                m.weight.data.fill_(0.0)
        self.policy_model.apply(init_weights)

    def predict(self, s):
        """
            Compute the action probabilities of state s using the learning model.

            INPUTS:
                s: input state array
            
            OUTPUTS:
                estimated optimal action
        """

        return self.policy_model(torch.Tensor(s))

    def update_policy_model(self, returns, log_probs, deltas):
        """
            Update the weights of the policy network given the training samples.
            
            INPUTS:
                returns: return (cumulative rewards) for each step in an episode
                log_probs: log probability for each step
                deltas: delta (difference between R - R_bar + v(s') - v(s)) at each step in the path
        """

        policy_gradient = torch.zeros(log_probs.shape[0])
        for i, (log_prob, Gt) in enumerate(zip(log_probs, returns)):
            # policy_gradient[i] = -log_prob * self.gamma**((len(returns)-i) * self.dt) * Gt
            policy_gradient[i] = -log_prob * self.gamma**((len(returns)-i) * self.dt) * deltas[i]

        loss = policy_gradient.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s, num_samples=1):
        """
            Estimate the policy distribution and sample an action, compute its log probability.
            
            INPUTS:
                s: Input state. Should be a 1D array.
            OUTPUTS:
                the selected action and log probability.
        """

        probs = self.predict(s)
        action_indices = torch.multinomial(probs, num_samples).item()
        actions = self.all_actions[action_indices].item()
        log_probs = torch.log(probs[action_indices])
        return actions, log_probs

    def update_w_hat(self):
        x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], self.w_hat_batch_size, replace=False)
        x_batch = self.dynamics_model.X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
        phi_x_batch = self.dynamics_model.phi(x_batch) # (phi_dim, w_hat_batch_size)

        with torch.no_grad():
            pi_response = self.predict(x_batch.T).T # (all_actions.shape[0], w_hat_batch_size)

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

        self.w_hat = OLS(
            (phi_x_batch - ((self.gamma**self.dt)*expectation_term_1)).T,
            expectation_term_2.T
        )

    def Q(self, x, u):
        return (-self.cost(x, u) + (self.gamma**self.dt)*self.w_hat.T @ self.dynamics_model.phi_f(x, u))[0,0]

    def actor_critic(
        self,
        num_training_episodes,
        num_steps_per_episode,
        R_learning_rate=0.003
    ):
        """
            REINFORCE algorithm
                
            INPUTS:
                num_training_episodes: number of episodes to train for
                num_steps_per_episode: number of steps per episode
        """

        # Initialize R_bar (Average reward)
        R_bar = 0.0

        # Initialize S
        initial_states = np.random.uniform(
            self.state_minimums,
            self.state_maximums,
            [self.dynamics_model.x_dim, num_training_episodes]
        )
        total_reward_episode = torch.zeros(num_training_episodes)

        for episode in range(num_training_episodes):
            states = torch.zeros([num_steps_per_episode, self.dynamics_model.x_dim])
            actions = torch.zeros(num_steps_per_episode)
            log_probs = torch.zeros(num_steps_per_episode)
            rewards = torch.zeros(num_steps_per_episode)
            deltas = torch.zeros(num_steps_per_episode)

            state = np.vstack(initial_states[:, episode])
            for step in range(num_steps_per_episode):
                # Add newest state to list of previous states
                states[step] = torch.Tensor(state)[:, 0]

                # A ~ π( . | S, θ )
                u, log_prob = self.get_action(state[:, 0])
                action = np.array([[u]])

                actions[step] = u
                log_probs[step] = log_prob

                # Take action A, observe S', R
                next_state = self.true_dynamics(state, action)
                curr_reward = -self.cost(state, action)[0,0]

                rewards[step] = curr_reward
                total_reward_episode[episode] += self.gamma**(step*self.dt) * curr_reward

                # Update δ
                delta = curr_reward - R_bar + (self.gamma**self.dt)*(self.w_hat.T @ self.phi(next_state)) - (self.w_hat.T @ self.phi(state))
                deltas[step] = torch.Tensor(delta)

                # Update R bar
                R_bar = R_bar + R_learning_rate * delta

                # Update state for next loop
                state = next_state

            returns = torch.zeros(num_steps_per_episode)
            for i in range(num_steps_per_episode-1, -1, -1):
                Q_val = self.Q(
                    np.vstack(states[i]),
                    np.array([[actions[i]]])
                )
                returns[i] = Q_val

            returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

            self.update_policy_model(returns, log_probs, deltas)
            if (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self.policy_model, self.saved_file_path)
                torch.save(torch.Tensor(self.w_hat), self.saved_file_path_w_hat)

            self.update_w_hat()