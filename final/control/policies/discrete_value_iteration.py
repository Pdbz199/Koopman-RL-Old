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
        save_data_path,
        use_ols=True,
        learning_rate=0.003,
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
                save_data_path - The path to save the training data and policy model.
                use_ols - Boolean to indicate whether or not to use OLS in computing new value function weights,
                learning_rate - The learning rate of the policy.
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
        self.save_data_path = save_data_path
        self.use_ols = use_ols
        self.learning_rate = learning_rate
        self.dt = dt

        self.discount_factor = self.gamma**self.dt

        if load_model:
            self.value_function_weights = torch.load(f"{self.save_data_path}/policy.pt")
        else:
            if self.use_ols:
                self.value_function_weights = torch.zeros((self.dynamics_model.phi_dim, 1))
            else:
                self.value_function_weights = torch.zeros((self.dynamics_model.phi_dim, 1), requires_grad=True)

        if not self.use_ols:
            self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)

    def pis(self, xs):
        """
            A distribution of probabilities for a given set of states.

            INPUTS:
                xs - 2D array of state column vectors.
            
            OUTPUTS:
                A 2D array of action probability column vectors.
        """

        # Compute phi(x) for each x
        phi_xs = self.dynamics_model.phi(xs)

        # Compute phi(x') for all ( phi(x), action ) pairs and compute V(x')s
        K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
        phi_x_primes = np.zeros([self.all_actions.shape[1], self.dynamics_model.phi_dim, xs.shape[1]])
        V_x_primes = torch.zeros([self.all_actions.shape[1], xs.shape[1]])
        for action_index in range(K_us.shape[0]):
            phi_x_primes_hat = K_us[action_index] @ phi_xs # (dim_phi, batch_size)
            x_primes_hat = self.dynamics_model.B.T @ phi_x_primes_hat # (X.shape[0], batch_size)
            # phi_x_primes[action_index] = phi_x_prime_hat
            phi_x_primes[action_index] = self.dynamics_model.phi(x_primes_hat) # (dim_phi, batch_size)
            V_x_primes[action_index] = self.V_phi_x(phi_x_primes[action_index]) # (1, batch_size)

        # Get costs indexed by the action and the state
        costs = torch.Tensor(self.cost(xs, self.all_actions)) # (all_actions.shape[1], batch_size)

        # Compute policy distribution
        inner_pi_us_values = -(costs + self.discount_factor*V_x_primes) # (all_actions.shape[1], xs.shape[1])
        inner_pi_us = inner_pi_us_values / self.regularization_lambda # (all_actions.shape[1], xs.shape[1])
        real_inner_pi_us = torch.real(inner_pi_us) # (all_actions.shape[1], xs.shape[1])

        # Max trick
        max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0) # xs.shape[1]
        diff = real_inner_pi_us - max_inner_pi_u

        delta = np.finfo(np.float32).eps # 1.1920929e-07
        # delta = np.finfo(np.float64).eps # 2.220446049250313e-16
        pi_us = torch.exp(diff) + delta # (all_actions.shape[1], xs.shape[1])
        Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]

        return pi_us / Z_x # (all_actions.shape[1], xs.shape[1])

    def V_phi_x(self, phi_x):
        """
            Compute V(phi_x).

            INPUTS:
                phi_x - Column vector of the observable of the state.

            OUTPUTS:
                Float value.
        """

        return self.value_function_weights.T @ torch.Tensor(phi_x)

    def V_x(self, x):
        """
            Compute V(x).

            INPUTS:
                x - Column vector of the state.

            OUTPUTS:
                Float value.
        """

        return self.V_phi_x(self.dynamics_model.phi(x))

    def discrete_bellman_error(self, batch_size):
        """
            Equation 12 in writeup.

            INPUTS:
                batch_size - Number of samples of the state space used to compute the Bellman error.

            OUTPUTS:
                Bellman error as described in our writeup.
        """

        # Get random sample of xs and phi(x)s from dataset
        x_batch_indices = np.random.choice(
            self.dynamics_model.X.shape[1],
            batch_size,
            replace=False
        )
        x_batch = self.dynamics_model.X[:, x_batch_indices] # (X.shape[0], batch_size)
        phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices] # (dim_phi, batch_size)

        # Compute V(x) for all phi(x)s
        V_xs = self.V_phi_x(phi_x_batch) # (1, batch_size)

        # Get costs indexed by the action and the state
        costs = torch.Tensor(self.cost(x_batch, self.all_actions)) # (all_actions.shape[1], batch_size)

        # Compute phi(x') for all ( phi(x), action ) pairs and compute V(x')s
        K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
        phi_x_primes = np.zeros([self.all_actions.shape[1], self.dynamics_model.phi_dim, batch_size])
        V_x_primes = torch.zeros([self.all_actions.shape[1], batch_size])
        for action_index in range(K_us.shape[0]):
            phi_x_primes_hat = K_us[action_index] @ phi_x_batch # (dim_phi, batch_size)
            x_primes_hat = self.dynamics_model.B.T @ phi_x_primes_hat # (X.shape[0], batch_size)
            # phi_x_primes[action_index] = phi_x_prime_hat
            phi_x_primes[action_index] = self.dynamics_model.phi(x_primes_hat) # (dim_phi, batch_size)
            V_x_primes[action_index] = self.V_phi_x(phi_x_primes[action_index]) # (1, batch_size)

        # Compute policy distribution
        inner_pi_us_values = -(costs + self.discount_factor*V_x_primes) # (all_actions.shape[1], batch_size)
        inner_pi_us = inner_pi_us_values / self.regularization_lambda # (all_actions.shape[1], batch_size)
        real_inner_pi_us = torch.real(inner_pi_us) # (all_actions.shape[1], batch_size)

        # Max trick
        max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0) # (batch_size,)
        diff = real_inner_pi_us - max_inner_pi_u # (all_actions.shape[1], batch_size)

        # Store a tiny delta in case we have 0s in exp()
        delta = np.finfo(np.float32).eps # 1.1920929e-07
        # delta = np.finfo(np.float64).eps # 2.220446049250313e-16

        # Softmax distribution
        pi_us = torch.exp(diff) + delta # (all_actions.shape[1], batch_size)
        Z_x = torch.sum(pi_us, axis=0) # (batch_size,)
        pis_response = pi_us / Z_x # (all_actions.shape[1], batch_size)

        # Compute log probabilities
        log_pis = torch.log(pis_response) # (all_actions.shape[1], batch_size)

        # Compute expectation
        expectation_u = torch.sum(
            (costs + \
                self.regularization_lambda*log_pis + \
                    self.discount_factor*V_x_primes) * pis_response,
            axis=0
        ).reshape(1, -1) # (1, batch_size)

        # Compute mean squared error
        squared_error = torch.pow(V_xs - expectation_u, 2) # (1, batch_size)
        mean_squared_error = torch.mean(squared_error) # scalar

        return mean_squared_error

    def get_action_and_log_prob(self, x, sample_size=None, is_greedy=False):
        """
            Compute the action given the current state.

            INPUTS:
                x - State of system as a column vector.
                sample_size - How many actions to sample. None gives 1 sample.

            OUTPUTS:
                Action from value iteration policy.
        """

        if sample_size is None:
            sample_size = self.dynamics_model.u_column_dim

        pis_response = (self.pis(x)[:, 0]).data.numpy()

        if is_greedy:
            selected_indices = np.ones(sample_size, dtype=np.int8) * np.argmax(pis_response)
        else:
            selected_indices = np.random.choice(
                np.arange(len(pis_response)),
                size=sample_size,
                p=pis_response
            )

        return (
            self.all_actions[0][selected_indices],
            np.log(pis_response[selected_indices])
        )

    def get_action(self, x, sample_size=None, is_greedy=False):
        """
            Compute the action given the current state.

            INPUTS:
                x - State of system as a column vector.
                sample_size - How many actions to sample. None gives 1 sample.

            OUTPUTS:
                Action from value iteration policy.
        """

        return self.get_action_and_log_prob(x, sample_size, is_greedy)[0]

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

        # Save original gamma and set gamma to first in array
        original_gamma = self.gamma
        if len(gammas) > 0:
            self.gamma = gammas[0]
        self.discount_factor = self.gamma**self.dt

        # Compute initial Bellman error
        BE = self.discrete_bellman_error(batch_size = batch_size * batch_scale).detach().numpy()
        bellman_errors = [BE]
        print(f"Initial Bellman error: {BE}")

        step = 0
        gamma_iteration_condition = self.gamma <= 0.99 or self.gamma == 1
        while gamma_iteration_condition:
            print(f"gamma for iteration #{step+1}: {self.gamma}")
            self.discount_factor = self.gamma**self.dt

            for epoch in range(training_epochs):
                # Get random batch of X and Phi_X from tensor training data
                x_batch_indices = np.random.choice(
                    self.dynamics_model.X.shape[1],
                    batch_size,
                    replace=False
                )
                x_batch = self.dynamics_model.X[:, x_batch_indices] # (X.shape[0], batch_size)
                phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices] # (dim_phi, batch_size)

                # Compute costs indexed by the action and the state
                costs = torch.Tensor(self.cost(x_batch, self.all_actions)) # (all_actions.shape[1], batch_size)

                # Compute V(x')s
                K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
                phi_x_primes = np.zeros((self.all_actions.shape[1], self.dynamics_model.phi_dim, batch_size))
                V_x_primes = torch.zeros((self.all_actions.shape[1], batch_size))
                for action_index in range(phi_x_primes.shape[0]):
                    phi_x_primes_hat = K_us[action_index] @ phi_x_batch # (phi_dim, batch_size)
                    x_primes_hat = self.dynamics_model.B.T @ phi_x_primes_hat # (X.shape[0], batch_size)
                    # phi_x_primes[action_index] = phi_x_prime_hat
                    phi_x_primes[action_index] = self.dynamics_model.phi(x_primes_hat) # (dim_phi, batch_size)
                    V_x_primes[action_index] = self.V_phi_x(phi_x_primes[action_index]) # (1, batch_size)

                # Compute policy distribution
                inner_pi_us_values = -(costs + self.discount_factor*V_x_primes) # (all_actions.shape[1], batch_size)
                inner_pi_us = inner_pi_us_values / self.regularization_lambda # (all_actions.shape[1], batch_size)
                real_inner_pi_us = torch.real(inner_pi_us) # (all_actions.shape[1], batch_size)

                # Max trick
                max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0) # (batch_size,)
                diff = real_inner_pi_us - max_inner_pi_u # (all_actions.shape[1], batch_size)

                # Store a tiny delta in case we have 0s in exp()
                delta = np.finfo(np.float32).eps # 1.1920929e-07
                # delta = np.finfo(np.float64).eps # 2.220446049250313e-16

                # Softmax distribution
                pi_us = torch.exp(diff) + delta # (all_actions.shape[1], batch_size)
                Z_x = torch.sum(pi_us, axis=0) # (batch_size,)
                pis_response = pi_us / Z_x # (all_actions.shape[1], batch_size)                

                # Compute log pi
                log_pis = torch.log(pis_response) # (all_actions.shape[1], batch_size)

                # Compute expectations
                expectation_term_1 = torch.sum(
                    (costs + \
                        self.regularization_lambda*log_pis + \
                            self.discount_factor*V_x_primes) * pis_response,
                    dim=0
                ).reshape(1, -1) # (1, batch_size)

                # Optimize value function weights
                if self.use_ols:
                    # OLS as in Lewis
                    self.value_function_weights = torch.linalg.lstsq(
                        torch.Tensor(phi_x_batch.T),
                        expectation_term_1.T
                    ).solution
                else:
                    # Compute loss
                    loss = torch.pow(V_x_primes - expectation_term_1, 2).mean()

                    # Backpropogation for value function weights
                    self.value_function_optimizer.zero_grad()
                    loss.backward()
                    self.value_function_optimizer.step()

                # Recompute Bellman error
                BE = self.discrete_bellman_error(batch_size = batch_size * batch_scale).detach().numpy()
                bellman_errors.append(BE)

                # Print epoch number
                print(f"Epoch number: {epoch+1}")

                # Every so often, print out and save the model weights and bellman errors
                # how_often_to_chkpt = 250
                how_often_to_chkpt = 20
                if (epoch+1) % how_often_to_chkpt == 0: # or True:
                    torch.save(self.value_function_weights, f"{self.save_data_path}/policy.pt")
                    np.save(f"{self.save_data_path}/training_data/bellman_errors.npy", np.array(bellman_errors))
                    print(f"Bellman error at epoch {epoch+1}: {BE}")

                # If the bellman error is less than or equal to epsilon, save the model weights and bellman errors
                if BE <= epsilon:
                    torch.save(self.value_function_weights, f"{self.save_data_path}/policy.pt")
                    np.save(f"{self.save_data_path}/training_data/bellman_errors.npy", np.array(bellman_errors))
                    print(f"Bellman error at epoch {epoch+1}: {BE}")
                    break

            step += 1

            if len(gammas) == 0 and gamma_increment_amount == 0:
                gamma_iteration_condition = False
                break

            if self.gamma == 0.99: break

            if len(gammas) > 0:
                self.gamma = gammas[step]
            else:
                self.gamma += gamma_increment_amount

            if self.gamma > 0.99: self.gamma = 0.99

            gamma_iteration_condition = self.gamma <= 0.99

        self.gamma = original_gamma
        self.discount_factor = self.gamma**self.dt