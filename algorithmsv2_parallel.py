import numpy as np
# import torch

from scipy.special import logsumexp
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
        self.w = np.load('bellman-weights.npy') if load else np.ones([self.Phi_X.shape[0],1], dtype=np.float64) # Default weights of 1s

        self.weight_regularization_bool = weight_regularization_bool #Bool for including weight regularization in Bellman loss functions
        self.weight_regularization_lambda = weight_regularization_lambda
        self.load = load
        self.optimizer = optimizer

    def inner_pi_us(self, us, xs):
        phi_x_primes = self.tensor.K_(us) @ self.phi(xs) # self.us.shape[1] x dim_phi x self.xs.shape[1]
        inner_pi_us = -(self.cost(xs, us).T + self.gamma * (self.w.T @ phi_x_primes)[:,0]) # self.us.shape[1] x self.xs.shape[1]
        return inner_pi_us*(1/self.weight_regularization_lambda)

    def pis(self, xs):
        delta = 1e-6
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
        
        
        
# def init_adam_states(feature_dim):
#     v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
#     s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
#     return ((v_w, s_w), (v_b, s_b))

# def adam(params, states, hyperparams):
#     beta1, beta2, eps = 0.9, 0.999, 1e-6
#     for p, (v, s) in zip(params, states):
#         with torch.no_grad():
#             v[:] = beta1 * v + (1 - beta1) * p.grad
#             s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
#             v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
#             s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
#             p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
#                                                        + eps)
#         p.grad.data.zero_()
#     hyperparams['t'] += 1