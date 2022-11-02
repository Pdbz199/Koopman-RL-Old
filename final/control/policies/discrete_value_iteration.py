import numpy as np
import torch

from final.tensor import KoopmanTensor

class DiscreteKoopmanValueIterationPolicy:
    def __init__(
        self,
        true_dynamics,
        gamma,
        regularization_lambda,
        dynamics_model: KoopmanTensor,
        all_actions,
        cost,
        saved_file_path,
        dt=1.0,
        seed=123,
        load_model=False
    ):
        """
            Constructor for the discrete value iteration policy class.

            INPUTS:
                true_dynamics - The true dynamics of the system.
                gamma - The discount factor of the system.
                regularization_lamb - The regularization parameter of the policy.
                dynamics_model - The trained Koopman tensor for the system.
                all_actions - The actions that the policy can take. Should be a single dimensional array.
                cost - The cost function of the system. Function must take in states and actions and return scalars.
                saved_file_path - The path to save the policy model.
                dt - The time step of the system.
                seed - Random seed for reproducibility.
                load_model - Boolean indicating whether or not to load a saved model.

            OUTPUTS:
                Instance of value iteration policy class.
        """

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.true_dynamics = true_dynamics
        self.gamma = gamma
        self.regularization_lambda = regularization_lambda
        self.dynamics_model = dynamics_model
        self.all_actions = all_actions
        self.cost = cost
        self.saved_file_path = saved_file_path
        self.dt = dt

        self.discount_factor = self.gamma**self.dt

        if load_model:
            self.value_function_weights = torch.load(self.saved_file_path)
        else:
            self.value_function_weights = torch.zeros([1, self.dynamics_model.phi_dim])

    def pis(self, xs):
        """
            A distribution of probabilities for a given set of states.

            INPUTS:
                xs - 2D array of state column vectors.
            
            OUTPUTS:
                A 2D array of action probability column vectors.
        """

        # TODO: Confirm `with torch.no_grad()` should be here
        with torch.no_grad():
            # Inner pi us
            us = self.all_actions

            phi_x_primes = self.dynamics_model.K_(us) @ self.dynamics_model.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]

            V_x_primes_arr = torch.zeros([self.all_actions.shape[1], xs.shape[1]])
            for u in range(phi_x_primes.shape[0]):
                V_x_primes_arr[u] = self.value_function_weights @ torch.Tensor(phi_x_primes[u]) # (1, xs.shape[1])

            inner_pi_us_values = -(torch.Tensor(self.cost(xs, us)) + self.discount_factor*V_x_primes_arr) # us.shape[1] x xs.shape[1]
            inner_pi_us = inner_pi_us_values * (1 / self.regularization_lambda) # us.shape[1] x xs.shape[1]
            inner_pi_us_response = torch.real(inner_pi_us) # all_actions.shape[1] x xs.shape[1]

            # TODO: Figure out why exponentiate action/state pair

            # Max trick
            max_inner_pi_u = torch.amax(inner_pi_us_response, axis=0) # xs.shape[1]
            diff = inner_pi_us_response - max_inner_pi_u

            delta = np.finfo(np.float32).eps # 1e-25
            pi_us = torch.exp(diff) + delta # all_actions.shape[1] x xs.shape[1]
            Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]
            
            return pi_us / Z_x # all_actions.shape[1] x xs.shape[1]

    def discrete_bellman_error(self, batch_size):
        """
            Equation 12 in writeup.

            INPUTS:
                batch_size - Number of samples of the state space used to compute the Bellman error.

            OUTPUTS:
                Bellman error as described in our writeup.
        """

        x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], batch_size, replace=False)
        x_batch = self.dynamics_model.X[:, x_batch_indices] # X.shape[0] x batch_size
        phi_xs = self.dynamics_model.Phi_X[:, x_batch_indices] # dim_phi x batch_size
        phi_x_primes = self.dynamics_model.K_(self.all_actions) @ phi_xs # all_actions.shape[1] x dim_phi x batch_size

        pis_response = self.pis(x_batch) # all_actions.shape[1] x x_batch_size
        log_pis = torch.log(pis_response) # all_actions.shape[1] x batch_size

        # Compute V(x)'s
        V_x_primes_arr = torch.zeros([self.all_actions.shape[1], batch_size])
        for u in range(phi_x_primes.shape[0]):
            V_x_primes_arr[u] = self.value_function_weights @ torch.Tensor(phi_x_primes[u])
        
        # Get costs
        costs = torch.Tensor(
            self.cost(x_batch, self.all_actions)
        ) # all_actions.shape[0] x batch_size

        # Compute expectations
        expectation_us = (costs + self.regularization_lambda*log_pis + self.discount_factor*V_x_primes_arr) * pis_response # all_actions.shape[1] x batch_size
        expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

        # Use model to get V(x) for all phi(x)s
        V_xs = self.value_function_weights @ torch.Tensor(phi_xs) # (batch_size, 1)

        # Compute squared differences
        squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
        # total = torch.sum(squared_differences) / batch_size # scalar
        total = torch.mean(squared_differences) # scalar

        return total # / torch.mean(torch.linalg.norm(torch.Tensor(phi_x_primes), dim=1))

    def get_action(self, x, sample_size=None):
        """
            Compute the action given the current state.

            INPUTS:
                x - State of system as a column vector.
                sample_size - How many actions to sample. None gives 1 sample.

            OUTPUTS:
                Action from value iteration policy.
        """

        pis_response = self.pis(x)[:,0]

        if sample_size is None:
            sample_size = self.dynamics_model.u_column_dim
        
        return np.random.choice(self.all_actions[0], size=sample_size, p=pis_response.data.numpy())

    def train(
        self,
        training_epochs,
        batch_size=2**12,
        batch_scale=1,
        epsilon=1e-2,
        gammas=[],
        gamma_increment_amount=0.0
    ):
        """
            Train the value iteration model. This updates the class parameters without returning anything.
            After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.

            INPUTS:
                training_epochs - Number of epochs for which to train the model.
                batch_size - Sample of states for computing the value function weights.
                batch_scale - Scale factor that is multiplied by batch_size for computing the Bellman error.
                epsilon - End the training process if the Bellman error < epsilon.
                gammas - Array of gammas to try in case of iterating on the discounting factors.
                gamma_increment_amount - Amount by which to increment gamma until it reaches 0.99. If 0.0, no incrementing.
        """

        original_gamma = self.gamma
        if len(gammas) > 0:
            self.gamma = gammas[0]
        self.discount_factor = self.gamma**self.dt
        print("gamma:", self.gamma)

        bellman_errors = [self.discrete_bellman_error(batch_size*batch_scale).data.numpy()]
        BE = bellman_errors[-1]
        print("Initial Bellman error:", BE)

        step = 0
        gammaIterationCondition = self.gamma <= 0.99
        while gammaIterationCondition:
            print("gamma:", self.gamma)
            self.discount_factor = self.gamma**self.dt

            for epoch in range(training_epochs):
                # TODO: Double check value function approximation here (can compare to `update_w_hat()` implementations)

                # Get random batch of X and Phi_X
                x_batch_indices = np.random.choice(self.dynamics_model.X.shape[1], batch_size, replace=False)
                x_batch = self.dynamics_model.X[:,x_batch_indices] # X.shape[0] x batch_size
                phi_x_batch = self.dynamics_model.Phi_X[:,x_batch_indices] # dim_phi x batch_size

                # Get current distribution of actions for each state
                pis_response = self.pis(x_batch) # (all_actions.shape[1], batch_size)
                log_pis = torch.log(pis_response) # (all_actions.shape[1], batch_size)

                # Compute V(x')s
                phi_x_primes = self.dynamics_model.K_(self.all_actions) @ phi_x_batch # all_actions.shape[1] x dim_phi x batch_size
                V_x_primes_arr = torch.zeros([self.all_actions.shape[1], batch_size])
                for u in range(phi_x_primes.shape[0]):
                    V_x_primes_arr[u] = self.value_function_weights @ torch.Tensor(phi_x_primes[u])

                # Get costs
                costs = torch.Tensor(
                    self.cost(x_batch, self.all_actions)
                ) # (all_actions.shape[1], batch_size)

                # Compute expectations
                expectation_term_1 = torch.sum(
                    torch.mul(
                        (costs + self.regularization_lambda*log_pis + self.discount_factor*V_x_primes_arr),
                        pis_response
                    ),
                    dim=0
                ).reshape(1,-1) # (1, batch_size)

                # OLS as in Lewis
                self.value_function_weights = torch.linalg.lstsq(
                    torch.Tensor(phi_x_batch.T),
                    expectation_term_1.T
                ).solution.T

                # Recompute Bellman error
                BE = self.discrete_bellman_error(batch_size*batch_scale).data.numpy()
                bellman_errors = np.append(bellman_errors, BE)

                # Every so often, print out and save the model weights and the bellman error(s)
                # Or, if the bellman error is less than the epsilon, save the model weights
                if epoch == 0 or (epoch+1) % 250 == 0:
                    # np.save('double_well_bellman_errors.npy', bellman_errors)
                    torch.save(self.value_function_weights, self.saved_file_path)
                    print(f"Bellman error at epoch {epoch+1}: {BE}")

                if BE <= epsilon:
                    torch.save(self.value_function_weights, self.saved_file_path)
                    break

            step += 1

            if len(gammas) == 0 and gamma_increment_amount == 0:
                gammaIterationCondition = False
                break

            if self.gamma == 0.99: break

            if len(gammas) > 0:
                self.gamma = gammas[step]
            else:
                self.gamma += gamma_increment_amount

            if self.gamma > 0.99: self.gamma = 0.99

            gammaIterationCondition = self.gamma <= 0.99

        self.gamma = original_gamma
        self.discount_factor = self.gamma**self.dt