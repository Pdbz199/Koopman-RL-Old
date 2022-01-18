import numpy as np

def rho(u, o='unif', a=0, b=1):
    if o == 'unif':
        return 1 / ( b - a )
    if o == 'normal':
        return np.exp( -u**2 / 2 ) / ( np.sqrt( 2 * np.pi ) )

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
        phi,
        psi,
        K_hat,
        cost,
        bellmanErrorType=0,
        learning_rate=1e-3,
        epsilon=1,
        weightRegularizationBool=1,
        weightRegLambda=1e-2,
        u_batch_size=50
    ):
        self.X = X # Collection of observations
        self.Phi_X = phi(X) # Collection of lifted observations
        self.All_U = All_U # U is a collection of all POSSIBLE actions as row vectors
        self.u_lower = u_bounds[0] # lower bound on actions in continuous case
        self.u_upper = u_bounds[1] # upper bound on actions in continuous case
        self.u_batch_size = u_batch_size
        self.phi = phi # Dictionary function for X
        self.psi = psi # Dictionary function for U
        self.K_hat = K_hat # Estimated Koopman Tensor
        self.cost = cost # Cost function to optimize
        self.bellmanErrorType = bellmanErrorType
        self.bellmanError = self.discreteBellmanError if bellmanErrorType == 0 else self.continuousBellmanError
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.w = np.ones([K_hat.shape[0],1]) # Default weights of 1s

        self.weightRegularization = weightRegularizationBool #Bool for including weight regularization in Bellman loss functions
        self.weightRegLambda = weightRegLambda

    def K_us(self, us):
        ''' Pick out Koopman operator given a matrix of action column vectors '''
        return np.einsum('ijz,zk->kij', self.K_hat, self.psi(us))

    def inner_pi_us(self, us, xs):
        #! look carefully through this because of removed loop
        phi_x_primes = self.K_us(us) @ self.phi(xs) # self.us.shape[1] x dim_phi x self.xs.shape[1]
        inner_pi_us = -(self.cost(xs, us).T + (self.w.T @ phi_x_primes)[:,0]) # self.us.shape[1] x self.xs.shape[1]
        return inner_pi_us

    def pis(self, xs):
        #! look carefully through this because of removed loop
        if self.bellmanErrorType == 0: # Discrete
            inner_pi_us = self.inner_pi_us(self.All_U, xs) # self.All_U.shape[1] x self.xs.shape[1]
            inner_pi_us = np.real(inner_pi_us) # self.All_U.shape[1] x self.xs.shape[1]
            max_inner_pi_u = np.amax(inner_pi_us, axis=0) # self.xs.shape[1]
            pi_us = np.exp(inner_pi_us - max_inner_pi_u) # self.All_U.shape[1] x self.xs.shape[1]
            Z_x = np.sum(pi_us, axis=0) # self.xs.shape[1]

            #normalization = (self.u_upper-self.u_lower)
            normalization = 1
            return pi_us / (Z_x*normalization) # self.All_U.shape[1] x self.xs.shape[1]
        else: # Continuous
            inner_pi_us = self.inner_pi_us(self.All_U, xs)
            inner_pi_us = np.real(inner_pi_us)
            max_inner_pi_u = np.max(inner_pi_us)
            pi_us = np.exp(inner_pi_us - max_inner_pi_u)
            Z_x = np.sum(pi_us)

            #Normalization follows from montecarlo integration approach to estimate true Z_x
            normalization = (self.u_upper-self.u_lower)/self.u_batch_size
            return pi_us / (Z_x*normalization)

    def discreteBellmanError(self):
        ''' Equation 12 in writeup '''

        phi_xs = self.phi(self.X) # dim_phi x self.X.shape[1]
        pis = self.pis(self.X) # self.All_U.shape[1] x self.X.shape[1]

        # pi_sum = np.sum(pis)
        # assert np.isclose(pi_sum, 1, rtol=1e-3, atol=1e-4)

        phi_x_primes = self.K_us(self.All_U) @ phi_xs # self.All_U.shape[1] x dim_phi x self.X.shape[1]
        weighted_phi_x_primes = (self.w.T @ phi_x_primes)[:,0] # self.All_U.shape[1] x self.X.shape[1]
        costs = self.cost(self.X, self.All_U).T # self.All_U.shape[1] x self.X.shape[1]
        expectation_us = (costs + np.log(pis) + weighted_phi_x_primes) * pis # self.All_U.shape[1] x self.X.shape[1]
        expectation_u = np.sum(expectation_us, axis=0) # self.X.shape[1]

        squared_differences = np.power((self.w.T @ phi_xs) - expectation_u, 2) # 1 x self.X.shape[1]
        total = np.sum(squared_differences) # scalar
                
        return total

    def algorithm2(self, batch_size):
        ''' Bellman error optimization '''

        BE = self.bellmanError()
        bellmanErrors = [BE]
        gradientNorms = []
        print("Initial Bellman error:", BE)

        if self.bellmanErrorType == 0: # if discrete BE
            n = 0
            while BE > self.epsilon:
                #! look carefully through this because of removed loop
                x_batch_indices = np.random.choice(self.X.shape[1], batch_size, replace=False)
                x_batch = self.X[:,x_batch_indices] # self.X.shape[0] x batch_size
                phi_x_batch = self.phi(x_batch) # dim_phi x batch_size

                weighted_phi_xs = (self.w.T @ phi_x_batch) # 1 x batch_size

                pis = np.vstack(self.pis(x_batch)) # self.All_U.shape[1] x batch_size
                log_pis = np.log(pis) # self.All_U.shape[1] x batch_size
                K_us = self.K_us(self.All_U) # self.All_U.shape[1] x dim_phi x dim_phi
                phi_x_primes = K_us @ phi_x_batch # self.All_U.shape[1] x dim_phi x batch_size
                weighted_phi_x_primes = (self.w.T @ phi_x_primes)[:,0] # self.All_U.shape[1] x batch_size
                costs = self.cost(x_batch, self.All_U).T # self.All_U.shape[1] x batch_size
                costs_plus_log_pis = costs + log_pis # self.All_U.shape[1] x batch_size

                expectationTerm1 = np.sum((costs_plus_log_pis + weighted_phi_x_primes) * pis, axis=0) # batch_size
                # (pis.T @ phi_x_primes) 64x100 x 100x6x64
                # with loop you get 1x100 x 100x6 => 1x6
                # expectationTerm2 = (pis.T @ phi_x_primes).T #! Error: size 6 is different from 100
                expectationTerm2 = np.einsum('ux,upx->px', pis, phi_x_primes) # dim_phi x batch_size

                # Equations 22/23 in writeup
                # print((weighted_phi_xs - expectationTerm1).shape) # 1 x batch_size
                # print((phi_x_batch - expectationTerm2).shape) # dim_phi x batch_size
                # print(((weighted_phi_xs - expectationTerm1) * (phi_x_batch - expectationTerm2)).shape) # dim_phi x batch_size
                difference = ((weighted_phi_xs - expectationTerm1) * (phi_x_batch - expectationTerm2)) / batch_size # dim_phi x batch_size
                nabla_w = np.sum(difference, axis=1) # dim_phi
                nabla_w = nabla_w.reshape((phi_x_batch.shape[0],1)) # dim_phi x 1

                gradientNorm = l2_norm(nabla_w, np.zeros_like(nabla_w)) # scalar
                gradientNorms.append(gradientNorm)

                # Update weights
                assert self.w.shape == nabla_w.shape
                self.w = self.w - (self.learning_rate * nabla_w)

                # Recompute Bellman error
                BE = self.bellmanError()
                bellmanErrors.append(BE)
                n += 1
                print("Current Bellman error:", BE)
                # if not n%100:
                #     np.save('bellman-weights.npy', self.w)
                #     print("Current Bellman error:", BE)

            return bellmanErrors, gradientNorms