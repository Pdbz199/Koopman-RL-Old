import numpy as np
import torch

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS, SINDy
# sys.path.append('../../')
# import observables

class ContinuousKoopmanPolicyIterationPolicy:
    """
        Compute the optimal policy for the given state using continuous Koopman policy iteration methodology.
    """

    def __init__(
        self,
        true_dynamics,
        gamma,
        lamb,
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
                lamb: The regularization parameter of the policy.
                dynamics_model: The trained Koopman tensor for the system.
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
        self.lamb = lamb
        self.dynamics_model = dynamics_model
        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.all_actions = all_actions
        self.cost = cost
        self.saved_file_path = saved_file_path
        self.dt = dt
        self.learning_rate = learning_rate
        self.w_hat_batch_size = w_hat_batch_size

        self.max_phi_norm = torch.linalg.norm(
            torch.Tensor(np.max(self.dynamics_model.Phi_X, axis=1))
        )
        # print(f"Max phi norm: {self.max_phi_norm}")

        if load_model:
            saved_model = torch.load(self.saved_file_path)
            self.alpha = saved_model.alpha
            self.beta = saved_model.beta
            self.w_hat = saved_model.w_hat
        else:
            self.alpha = torch.zeros([1, self.dynamics_model.phi_dim], requires_grad=True)
            # self.alpha = torch.zeros([1, self.dynamics_model.x_dim+1], requires_grad=True)
            self.beta = torch.tensor(0.1, requires_grad=True)
            self.w_hat = np.zeros(self.dynamics_model.phi_column_dim)

        self.optimizer = torch.optim.Adam([self.alpha, self.beta], self.learning_rate)

    def get_action_distribution(self, s):
        phi_s = torch.Tensor(self.dynamics_model.phi(s))
        # phi_s = torch.cat([torch.Tensor(s), torch.tensor([[1]])])
        mu = (self.alpha @ phi_s)[0,0]
        # mu = (self.alpha @ (phi_s / self.max_phi_norm))[0,0]
        sigma = torch.exp(self.beta)
        return torch.distributions.normal.Normal(mu, sigma, validate_args=False)

    def get_action(self, s):
        """
            Estimate the policy and sample an action, compute its log probability
            @param s: input state (column vector)
            @return: the selected action and log probability
        """
        action_distribution = self.get_action_distribution(s)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action, log_prob

    def update_w_hat(self):
        x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], self.w_hat_batch_size, replace=False)
        x_batch = self.dynamics_model.X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
        phi_x_batch = self.dynamics_model.phi(x_batch) # (phi_dim, w_hat_batch_size)

        with torch.no_grad():
            pi_response = np.zeros([self.all_actions.shape[0],self.w_hat_batch_size])
            for state_index, state in enumerate(x_batch.T):
                action_distribution = self.get_action_distribution(state.reshape(-1,1))
                pi_response[:, state_index] = action_distribution.log_prob(torch.tensor(self.all_actions))

        phi_x_prime_batch = self.dynamics_model.K_(np.array([self.all_actions])) @ phi_x_batch # (all_actions.shape[0], phi_dim, w_hat_batch_size)
        phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response) # (all_actions.shape[0], phi_dim, w_hat_batch_size)
        expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

        reward_batch_prob = np.einsum('uw,uw->wu', -self.cost(x_batch, np.array([self.all_actions])), pi_response) # (w_hat_batch_size, all_actions.shape[0])
        expectation_term_2 = np.array([
            np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
        ]) # (1, w_hat_batch_size)

        self.w_hat = OLS(
            (phi_x_batch - ((self.gamma**self.dt)*expectation_term_1)).T,
            expectation_term_2.T
        )

    def Q(self, x, u):
        V_x_prime = (self.gamma**self.dt)*self.w_hat.T @ self.dynamics_model.phi_f(x, u)
        return (-self.cost(x, u) + V_x_prime)[0,0]

    def update_policy_model(self, returns, log_probs):
        """
            Update the weights of the policy network given the training samples.
            
            INPUTS:
                returns: return (cumulative rewards) for each step in an episode
                log_probs: log probability for each step
        """

        policy_gradient = torch.zeros(log_probs.shape[0])
        for i, (log_prob, Gt) in enumerate(zip(log_probs, returns)):
            policy_gradient[i] = -log_prob * self.gamma**((len(returns)-i) * self.dt) * Gt

        # loss = policy_gradient.sum()
        loss = policy_gradient.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reinforce(self, num_training_episodes, num_steps_per_episode):
        """
            REINFORCE algorithm
                
            INPUTS:
                num_training_episodes - Number of episodes to train for.
                num_steps_per_episode - Number of steps per episode.
        """

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

            state = np.vstack(initial_states[:, episode])
            for step in range(num_steps_per_episode):
                states[step] = torch.Tensor(state)[:, 0]

                u, log_prob = self.get_action(state)
                action = np.array([[u]])
                actions[step] = u
                log_probs[step] = log_prob

                rewards[step] = -self.cost(state, action)[0,0]
                total_reward_episode[episode] += self.gamma**(step*self.dt) * rewards[step]

                state = self.true_dynamics(state, action)

            returns = torch.zeros(num_steps_per_episode)
            for i in range(num_steps_per_episode-1, -1, -1):
                Q_val = self.Q(
                    np.vstack(states[i]),
                    np.array([[actions[i]]])
                )
                returns[i] = Q_val

            returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

            self.update_policy_model(returns, log_probs)
            if (episode+1) % 250 == 0:
                print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                torch.save(self, self.saved_file_path)

            self.update_w_hat()