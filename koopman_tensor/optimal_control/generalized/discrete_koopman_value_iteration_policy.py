from random import sample
import numpy as np
import torch
import torch.nn as nn

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS

class DiscreteKoopmanValueIterationPolicy:
    """
        Compute the optimal policy for the given state using discrete Koopman value iteration methodology.
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
        learning_rate=0.003
    ):
        self.true_dynamics = true_dynamics
        self.gamma = gamma
        self.lamb = lamb
        self.dynamics_model = dynamics_model
        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.all_actions = all_actions
        self.cost = cost
        self.saved_file_path = saved_file_path
        self.learning_rate = learning_rate

        self.policy_model = torch.nn.Sequential(torch.nn.Linear(self.dynamics_model.phi_dim, 1))
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), self.learning_rate)

        def init_weights(m):
            if type(m) == torch.nn.Linear:
                m.weight.data.fill_(0.0)
        self.policy_model.apply(init_weights)

    def inner_pi_us(self, us, xs):
        phi_x_primes = self.dynamics_model.K_(us) @ self.dynamics_model.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]

        V_x_primes_arr = torch.zeros([self.all_actions.shape[0], xs.shape[1]])
        for u in range(phi_x_primes.shape[0]):
            V_x_primes_arr[u] = self.policy_model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

        inner_pi_us_values = -(torch.from_numpy(self.cost(xs, us)).float() + self.gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

        return inner_pi_us_values * (1 / self.lamb) # us.shape[1] x xs.shape[1]

    def pis(self, xs):
        delta = np.finfo(np.float32).eps # 1e-25

        inner_pi_us_response = torch.real(self.inner_pi_us(np.array([self.all_actions]), xs)) # all_actions.shape[0] x xs.shape[1]

        # Max trick
        max_inner_pi_u = torch.amax(inner_pi_us_response, axis=0) # xs.shape[1]
        diff = inner_pi_us_response - max_inner_pi_u

        pi_us = torch.exp(diff) + delta # all_actions.shape[0] x xs.shape[1]
        Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]
        
        return pi_us / Z_x # all_actions.shape[0] x xs.shape[1]

    def discrete_bellman_error(self, batch_size):
        """ Equation 12 in writeup """
        x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], batch_size, replace=False)
        x_batch = self.dynamics_model.X[:, x_batch_indices] # X.shape[0] x batch_size
        phi_xs = self.dynamics_model.phi(x_batch) # dim_phi x batch_size
        phi_x_primes = self.dynamics_model.K_(np.array([self.all_actions])) @ phi_xs # all_actions.shape[0] x dim_phi x batch_size

        pis_response = self.pis(x_batch) # all_actions.shape[0] x x_batch_size
        log_pis = torch.log(pis_response) # all_actions.shape[0] x batch_size

        # Compute V(x)'s
        V_x_primes_arr = torch.zeros([self.all_actions.shape[0], batch_size])
        for u in range(phi_x_primes.shape[0]):
            V_x_primes_arr[u] = self.policy_model(torch.from_numpy(phi_x_primes[u].T).float()).T
        
        # Get costs
        costs = torch.from_numpy(self.cost(x_batch, np.array([self.all_actions]))).float() # all_actions.shape[0] x batch_size

        # Compute expectations
        expectation_us = (costs + self.lamb*log_pis + self.gamma*V_x_primes_arr) * pis_response # all_actions.shape[0] x batch_size
        expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

        # Use model to get V(x) for all phi(x)s
        V_xs = self.policy_model(torch.from_numpy(phi_xs.T).float()) # (batch_size, 1)

        # Compute squared differences
        squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
        total = torch.sum(squared_differences) / batch_size # scalar

        return total

    def get_action(self, x, sample_size=None):
        with torch.no_grad():
            pis_response = self.pis(x)[:,0]

        if sample_size is None:
            sample_size = self.dynamics_model.u_column_dim
        return np.random.choice(self.all_actions, size=sample_size, p=pis_response.data.numpy())

    def train(self, training_epochs, batch_size, batch_scale, epsilon, gamma_increment_amount=0.02):
        bellman_errors = [self.discrete_bellman_error(batch_size*batch_scale).data.numpy()]
        BE = bellman_errors[-1]
        print("Initial Bellman error:", BE)

        while self.gamma <= 0.99:
            for epoch in range(training_epochs):
                # Get random batch of X and Phi_X
                x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], batch_size, replace=False)
                x_batch = self.dynamics_model.X[:,x_batch_indices] # X.shape[0] x batch_size
                phi_x_batch = self.dynamics_model.phi(x_batch) # dim_phi x batch_size

                # Compute estimate of V(x) given the current model
                V_x = self.policy_model(torch.from_numpy(phi_x_batch.T).float()).T # (1, batch_size)

                # Get current distribution of actions for each state
                pis_response = self.pis(x_batch) # (all_actions.shape[0], batch_size)
                log_pis = torch.log(pis_response) # (all_actions.shape[0], batch_size)

                # Compute V(x)'
                phi_x_primes = self.dynamics_model.K_(np.array([self.all_actions])) @ phi_x_batch # all_actions.shape[0] x dim_phi x batch_size
                V_x_primes_arr = torch.zeros([self.all_actions.shape[0], batch_size])
                for u in range(phi_x_primes.shape[0]):
                    V_x_primes_arr[u] = self.policy_model(torch.from_numpy(phi_x_primes[u].T).float()).T

                # Get costs
                costs = torch.from_numpy(self.cost(x_batch, np.array([self.all_actions]))).float() # (all_actions.shape[0], batch_size)

                # Compute expectations
                expectation_term_1 = torch.sum(
                    torch.mul(
                        (costs + self.lamb*log_pis + self.gamma*V_x_primes_arr),
                        pis_response
                    ),
                    dim=0
                ).reshape(1,-1) # (1, batch_size)

                # Equation 2.21 in Overleaf
                loss = torch.sum( torch.pow( V_x - expectation_term_1, 2 ) ) # ()
                
                # Back propogation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Recompute Bellman error
                BE = self.discrete_bellman_error(batch_size*batch_scale).data.numpy()
                bellman_errors = np.append(bellman_errors, BE)

                # Every so often, print out and save the bellman error(s)
                if (epoch+1) % 250 == 0:
                    # np.save('double_well_bellman_errors.npy', bellman_errors)
                    torch.save(self.policy_model, self.saved_file_path)
                    print(f"Bellman error at epoch {epoch+1}: {BE}")

                if BE <= epsilon:
                    torch.save(self.policy_model, self.saved_file_path)
                    break

            self.gamma += gamma_increment_amount