import numpy as np
# import torch

# from scipy.special import logsumexp
# from torch import logsumexp

# def rho(u, o='unif', a=0, b=1):
#     if o == 'unif':
#         return 1 / ( b - a )
#     if o == 'normal':
#         return np.exp( -u**2 / 2 ) / ( np.sqrt( 2 * np.pi ) )

def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        # print("Shape 1:", true_state.shape)
        # print("Shape 2:", predicted_state.shape)
        raise Exception('The dimensions of the parameters did not match and therefore cannot be compared.')
    
    err = true_state - predicted_state
    return np.sum(np.power(err, 2))

class algos:
    def __init__(
        self,
        X,
        All_U,
        u_bounds,
        tensor,
        cost,
        beta=0.9,
        beta2=0.999,
        gamma=1.0,
        bellman_error_type=0,
        learning_rate=1e-2,
        epsilon=1,
        weight_regularization_bool=False,
        weight_regularization_lambda=1,
        u_batch_size=50,
        load=False,
        optimizer='sgd'
    ):
        self.X = X # Collection of observations
        self.Phi_X = tensor.phi(X) # Collection of lifted observations
        self.All_U = All_U # U is a collection of all POSSIBLE actions as row vectors
        self.u_lower = u_bounds[0] # lower bound on actions in continuous case
        self.u_upper = u_bounds[1] # upper bound on actions in continuous case
        self.u_batch_size = u_batch_size
        self.phi = tensor.phi # Dictionary function for X
        self.psi = tensor.psi # Dictionary function for U
        self.tensor = tensor
        self.cost = cost # Cost function to optimize
        self.beta = beta
        self.beta2 = beta2
        self.gamma = gamma
        self.bellman_error_type = bellman_error_type
        self.bellman_error = self.discrete_bellman_error if bellman_error_type == 0 else self.continuous_bellman_error
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.w = np.load('bellman-weights.npy') if load else np.zeros([self.Phi_X.shape[0],1], dtype=np.float64) # Default weights of 0s

        self.weight_regularization_bool = weight_regularization_bool #Bool for including weight regularization in Bellman loss functions
        self.weight_regularization_lambda = weight_regularization_lambda
        self.load = load
        self.optimizer = optimizer

    def inner_pi_us(self, us, xs):
        phi_x_primes = self.tensor.K_(us) @ self.phi(xs) # self.us.shape[1] x dim_phi x self.xs.shape[1]
        inner_pi_us = -(self.cost(xs, us).T + self.gamma * (self.w.T @ phi_x_primes)[:,0]) # self.us.shape[1] x self.xs.shape[1]
        return inner_pi_us*(1/self.weight_regularization_lambda)

    def pis(self, xs):
        delta = 1e-25
        if self.bellman_error_type == 0: # Discrete
            inner_pi_us = self.inner_pi_us(self.All_U, xs) # self.All_U.shape[1] x self.xs.shape[1]
            inner_pi_us = np.real(inner_pi_us) # self.All_U.shape[1] x self.xs.shape[1]
            max_inner_pi_u = np.amax(inner_pi_us, axis=0) # self.xs.shape[1]
            # min_inner_pi_u = np.amin(inner_pi_us, axis=0) # self.xs.shape[1]
            diff = inner_pi_us - max_inner_pi_u
            pi_us = np.exp(diff) + delta # self.All_U.shape[1] x self.xs.shape[1]
            Z_x = np.sum(pi_us, axis=0) # self.xs.shape[1]
            

            # normalization = (self.u_upper-self.u_lower)
            normalization = 1
            return [inner_pi_us, diff, pi_us / (Z_x * normalization)] # self.All_U.shape[1] x self.xs.shape[1]
            
        else: # Continuous
            inner_pi_us = self.inner_pi_us(self.All_U, xs)
            inner_pi_us = np.real(inner_pi_us)
            max_inner_pi_u = np.max(inner_pi_us)
            pi_us = np.exp(inner_pi_us - max_inner_pi_u)
            Z_x = np.sum(pi_us)

            #Normalization follows from montecarlo integration approach to estimate true Z_x
            normalization = (self.u_upper-self.u_lower)/self.u_batch_size
            return pi_us / (Z_x*normalization)

    def discrete_bellman_error(self, batch_size):
        ''' Equation 12 in writeup '''
        x_batch_indices = np.random.choice(self.X.shape[1], batch_size, replace=False)
        x_batch = self.X[:, x_batch_indices] # self.X.shape[0] x batch_size
        phi_xs = self.phi(x_batch) # dim_phi x batch_size
        pis_response = self.pis(x_batch)
        pis = pis_response[2] # self.All_U.shape[1] x batch_size
        # log_pis = pis_response[0] - logsumexp(pis_response[1], axis=0) # logsumexp(torch.from_numpy(pis_response[1]), dim=0).detach().cpu().numpy()
        #! check the axis and broadcasting for the logsumexp method of calculating log_pis
        log_pis = np.log(pis)

        # pi_sum = np.sum(pis)
        # assert np.isclose(pi_sum, 1, rtol=1e-3, atol=1e-4)

        phi_x_primes = self.tensor.K_(self.All_U) @ phi_xs # self.All_U.shape[1] x dim_phi x batch_size
        weighted_phi_x_primes = (self.w.T @ phi_x_primes)[:,0] # self.All_U.shape[1] x batch_size
        costs = self.cost(x_batch, self.All_U).T # self.All_U.shape[1] x batch_size
        expectation_us = (costs + self.weight_regularization_lambda*log_pis + self.gamma * weighted_phi_x_primes) * pis # self.All_U.shape[1] x batch_size
        expectation_u = np.sum(expectation_us, axis=0) # batch_size

        squared_differences = np.power((self.w.T @ phi_xs) - expectation_u, 2) # 1 x batch_size
        total = np.sum(squared_differences) / batch_size # scalar
                
        return total

    def algorithm2(self, batch_size):
        ''' Bellman error optimization '''

        bellman_errors = np.load('bellman_errors.npy') if self.load else np.array([self.bellman_error(batch_size*3)])
        BE = bellman_errors[-1]
        gradient_norms = np.load('gradient_norms.npy') if self.load else np.array([])
        print("Initial Bellman error:", BE)

        if self.bellman_error_type == 0: # if discrete BE
            n = np.load('n.npy') if self.load else 0
            nabla_w = 0
            momentum = np.load('momentum.npy') if self.load else 0
            second_moment = np.load('second-moment.npy') if self.load else 0
            small_num = 1e-6
            while BE > self.epsilon:
                x_batch_indices = np.random.choice(self.X.shape[1], batch_size, replace=False)
                x_batch = self.X[:,x_batch_indices] # self.X.shape[0] x batch_size
                phi_x_batch = self.phi(x_batch) # dim_phi x batch_size

                weighted_phi_x_batch = (self.w.T @ phi_x_batch) # 1 x batch_size

                pis_response = self.pis(x_batch)
                pis = np.vstack(pis_response[2]) # self.All_U.shape[1] x batch_size
                # log_pis = pis_response[0] - logsumexp(pis_response[1], axis=0) # logsumexp(torch.from_numpy(pis_response[1]), dim=0).detach().cpu().numpy() # self.All_U.shape[1] x batch_size
                log_pis = np.log(pis)
                K_us = self.tensor.K_(self.All_U) # self.All_U.shape[1] x dim_phi x dim_phi
                phi_x_primes = K_us @ phi_x_batch # self.All_U.shape[1] x dim_phi x batch_size
                weighted_phi_x_primes = (self.w.T @ phi_x_primes)[:,0] # self.All_U.shape[1] x batch_size
                costs = self.cost(x_batch, self.All_U).T # self.All_U.shape[1] x batch_size

                expectation_term_1 = np.sum((costs + self.weight_regularization_lambda*log_pis + self.gamma * weighted_phi_x_primes) * pis, axis=0) # batch_size
                expectation_term_2 = np.einsum('ux,upx->px', pis, self.gamma * phi_x_primes) # dim_phi x batch_size

                # Equations 22/23 in writeup
                difference = ((weighted_phi_x_batch - expectation_term_1) * (phi_x_batch - expectation_term_2)) # dim_phi x batch_size
                if self.optimizer == 'sgd': # traditional SGD
                    nabla_w = np.sum(difference, axis=1).reshape((phi_x_batch.shape[0],1)) # dim_phi x 1
                elif self.optimizer == 'sgdwm': # SGD w/ momentum
                    nabla_w = self.beta*nabla_w + np.sum(difference, axis=1).reshape((phi_x_batch.shape[0],1)) # dim_phi x 1
                elif self.optimizer == 'adam': # Adam
                    nabla_w = np.sum(difference, axis=1).reshape((phi_x_batch.shape[0],1))
                    momentum = self.beta*momentum + (1-self.beta)*nabla_w
                    second_moment = self.beta2*second_moment + (1-self.beta2) * nabla_w**2
                    mom_normalized = momentum/(1-self.beta**(n+1))
                    sec_moment_normalized = second_moment/(1-self.beta2**(n+1))
                    nabla_w = mom_normalized/(np.sqrt(sec_moment_normalized)+small_num)

                gradient_norm = l2_norm(nabla_w, np.zeros_like(nabla_w)) # scalar
                gradient_norms = np.append(gradient_norms, gradient_norm)

                # Update weights
                assert self.w.shape == nabla_w.shape
                self.w = self.w - (self.learning_rate * nabla_w)

                # Recompute Bellman error
                BE = self.bellman_error(batch_size*3)
                bellman_errors = np.append(bellman_errors, BE)
                n += 1

                if n%25 == 0:
                    np.save('bellman_errors.npy', bellman_errors)
                    np.save('gradient_norms.npy', gradient_norms)
                    np.save('bellman-weights.npy', self.w)
                    np.save('momentum.npy', momentum)
                    np.save('second-moment.npy', second_moment)
                    np.save('n.npy', n)
                    print("Current Bellman error:", BE)

            return bellman_errors, gradient_norms


    def REINFORCE(self, f, reward, sigma, step_size=0.0001):
        """
            Monte-Carlo policy gradient algorithm

            INPUTS
            f: Function of system dynamics
            reward: Reward function that can support input arrays
            sigma: variance in action space
            step_size: How much to step in direction of gradient

            OUTPUTS
            theta: Parameters that optimize output of Gaussian policy N(mu(s), sigma^2)
        """

        theta = np.zeros([self.Phi_X.shape[0],1], dtype=np.float64)
        sigma_squared = np.power(sigma, 2)
        def pi(x, theta):
            return np.random.normal(self.phi(x).T @ theta, sigma_squared)

        num_episodes = 300 #! randomly chosen
        num_steps_per_episode = 200 #! randomly chosen
        state_range = 20 #! randomly chosen

        x_path = np.zeros([self.X.shape[0],num_steps_per_episode+1])
        u_path = np.zeros([1,num_steps_per_episode])
        for episode in range(num_episodes):
            x = np.random.rand(self.X.shape[0],1) * state_range * np.random.choice([-1,1]) # random initial state

            x_path[:,0] = x[:,0]
            for step in range(num_steps_per_episode):
                u = pi(x, theta)
                x = f(x, u) #! Use Koopman dynamics (?)
                x_path[:,step+1] = x[:,0]
                u_path[:,step] = u

            for step in range(num_steps_per_episode):
                u = np.vstack(u_path[:,step])
                x = np.vstack(x_path[:,step])
                phi_x = self.phi(x)
                G = np.sum(reward(x_path[:,step:-1], u_path[:,step:]))
                #! theta blows up which eventually makes grad log(pi_theta) +/- inf
                theta = theta + step_size * G * (((u - (phi_x.T @ theta)) * phi_x) / sigma_squared) # Using Gaussian policy

            #* Might want to use advantage instead of simple sum of rewards, but idk how to do that
        return theta