from os import replace
import auxiliaries as aux
import numpy as np
# import pymp
import scipy.integrate as integrate
import time

def rho(u, o='unif', a=0, b=1):
    if o == 'unif':
        return 1 / ( b - a )
    if o == 'normal':
        return np.exp( -u**2 / 2 ) / ( np.sqrt( 2 * np.pi ) )

def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        print("Shape 1:", true_state.shape)
        print("Shape 2:", predicted_state.shape)
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

    def K_u(self, u):
        ''' Pick out Koopman operator given an action '''
        return np.einsum('ijz,zk->ij', self.K_hat, self.psi(u))

    def K_us(self, us):
        ''' Pick out Koopman operator given a matrix of action column vectors '''
        return np.einsum('ijz,zk->kij', self.K_hat, self.psi(us))

    def inner_pi_u(self, u, x):
        inner_pi_u = (-(self.cost(x, u) + self.w.T @ self.K_u(u) @ self.phi(x)))[0]
        return inner_pi_u

    def inner_pi_us(self, us, x):
        inner_pi_us = -(self.cost(x, us) + self.w.T @ self.K_us(us) @ self.phi(x)) #stack cols of x as cols in self.phi(x)
        return inner_pi_us[:,0,0]

    def pi_u(self, u, x):
        inner = self.inner_pi_u(u, x)
        return np.exp(inner)

    def discreteBellmanError(self):
        ''' Equation 12 in writeup '''

        total = 0
        # total = pymp.shared.array((1,1), dtype='float32')
        # with pymp.Parallel(8) as p:
        for i in range(0, self.X.shape[1]): # loop
            x = self.X[:,i].reshape(-1,1)
            # x = self.X[:, np.random.choice(np.arange(self.X.shape[1]))].reshape(-1,1)
            phi_x = self.phi(x)

            inner_pi_us = self.inner_pi_us(self.All_U, x) #vectorize
            inner_pi_us = np.real(inner_pi_us)
            max_inner_pi_u = np.max(inner_pi_us)
            pi_us = np.exp(inner_pi_us - max_inner_pi_u)
            Z_x = np.sum(pi_us)

            pis = pi_us / Z_x
            # pi_sum = np.sum(pis)
            # assert np.isclose(pi_sum, 1, rtol=1e-3, atol=1e-4)

            weighted_phi_x_primes = self.w.T @ self.K_us(self.All_U) @ phi_x
            expectation_us = (self.cost(x * np.ones([x.shape[0],2]), self.All_U) + np.log(pis) + weighted_phi_x_primes[:,0,0]) * pis
            expectation_u = np.sum(expectation_us)

            # with p.lock:
            total += np.power((self.w.T @ phi_x - expectation_u), 2)
                
        return total

    def continuousBellmanError(self):
        ''' Equation 3 in writeup modified for continuous action weight regularization added to help gradient explosion in Bellman algos '''
        total = 0
        self.All_U = np.random.uniform(self.u_lower, self.u_upper, [1,self.u_batch_size])
        # print(self.All_U.shape)
        for _ in range(int(self.X.shape[1]/100)): # loop
            # x = self.X[:,i].reshape(-1,1)
            x = self.X[:, np.random.choice(np.arange(self.X.shape[1]))].reshape(-1,1)
            phi_x = self.phi(x)

            inner_pi_us = self.inner_pi_us(self.All_U, x)
            inner_pi_us = np.real(inner_pi_us)
            max_inner_pi_u = np.max(inner_pi_us)
            pi_us = np.exp(inner_pi_us - max_inner_pi_u)
            Z_x = np.sum(pi_us)

            normalization = 4*self.u_batch_size # 4 for uniform dist on u and 20 for minibatch on u's
            pis = pi_us / (Z_x*normalization)
            # pi_sum = np.sum(pis)
            # assert np.isclose(pi_sum, 1, rtol=1e-3, atol=1e-4)

            weighted_phi_x_primes = self.w.T @ self.K_us(self.All_U) @ phi_x
            expectation_us = (self.cost(x * np.ones([x.shape[0],self.All_U.shape[1]]), self.All_U) + np.log(pis) + weighted_phi_x_primes[:,0,0]) * pis
            expectation_u = np.sum(expectation_us)

            total += np.power((self.w.T @ phi_x - expectation_u), 2)
        # pi = (lambda u, x, Z_x: np.exp(self.inner_pi_u(u, x)) / Z_x)
        # def expectation_u_integrand(u, x, phi_x, Z_x):
        #     pi_u_const = pi(np.array([[u]]), x, Z_x)
        #     return (self.cost(x, u) - np.log(pi_u_const) - self.w.T @ self.K_u(np.array([[u]])) @ phi_x) * pi_u_const

        # total = 0
        # for i in range(self.X.shape[1]):
        #     x = self.X[:,i].reshape(-1,1)
        #     phi_x = self.phi(x)

        #     Z_x = integrate.quad(np.exp(self.inner_pi_u), self.u_lower, self.u_upper, (x))[0]
        #     expectation_u = integrate.quad(expectation_u_integrand, self.u_lower, self.u_upper, (x, phi_x, Z_x))[0]

        #     total += np.power(( self.w.T @ phi_x - expectation_u ), 2)

        # #add weight regularization term to help with gradient explosion issues
        # # if self.weightRegularization:
        # #     total += self.weightRegLambda*(aux.l2_norm(self.w)**2)

        return total

    def algorithm2(self, batch_size):
        ''' Bellman error optimization '''

        BE = self.bellmanError()[0,0]
        bellmanErrors = [BE]
        gradientNorms = []
        print("Initial Bellman error:", BE)

        if not self.bellmanErrorType: # if discrete BE
            while BE > self.epsilon:
                x_batch_indices = np.random.choice(self.X.shape[1], batch_size, replace=False)
                x_batch = self.X[:,x_batch_indices]
                phi_x_batch = self.phi(x_batch)

                nabla_w = np.zeros_like(self.w)
                # nabla_w = pymp.shared.array(self.w.shape, dtype='float32')
                # with pymp.Parallel(8) as p:
                for i in range(0, x_batch.shape[1]):
                # for x1, phi_x1 in zip(x_batch.T, phi_x_batch.T): # loop
                    x = x_batch[:,i].reshape(-1,1)
                    phi_x = phi_x_batch[:,i].reshape(-1,1)

                    inner_pi_us = self.inner_pi_us(self.All_U, x)
                    inner_pi_us = np.real(inner_pi_us)
                    max_inner_pi_u = np.max(inner_pi_us)
                    pi_us = np.exp(inner_pi_us - max_inner_pi_u)
                    Z_x = np.sum(pi_us)

                    pis = pi_us / Z_x
                    log_pis = np.log(pis)
                    K_us = self.K_us(self.All_U)
                    costs = self.cost(
                        x * np.ones([x.shape[0],2]),
                        self.All_U
                    )
                    costs_plus_log_pis = costs + log_pis

                    expectationTerm1 = np.sum(pis.reshape(-1,1) * (costs_plus_log_pis.reshape(-1,1) + (self.w.T @ K_us @ phi_x).reshape(self.All_U.shape[1],1)))
                    expectationTerm2 = np.einsum('i, ijk -> jk', pis, K_us @ phi_x)

                    # Equation 13/14 in writeup
                    # with p.lock:
                    nabla_w += ((self.w.T @ phi_x - expectationTerm1) * (phi_x - expectationTerm2)) / batch_size

                gradientNorm = l2_norm(nabla_w, np.zeros_like(nabla_w))
                gradientNorms.append(gradientNorm)

                # Update weights
                assert self.w.shape == nabla_w.shape
                self.w = self.w - (self.learning_rate * nabla_w)
                # print("Current weights:", self.w)

                # Recompute Bellman error
                BE = self.bellmanError()[0,0]
                bellmanErrors.append(BE)
                print("Current Bellman error:", BE)

            return bellmanErrors, gradientNorms
        else:
            while BE > self.epsilon:
                x_batch_indices = np.random.choice(self.X.shape[1], batch_size, replace=False)
                x_batch = self.X[:,x_batch_indices]
                phi_x_batch = self.phi(x_batch)
                
                u1_batch = np.random.uniform(self.u_lower, self.u_upper, [1,self.u_batch_size])
                u2_batch = np.random.uniform(self.u_lower, self.u_upper, [1,self.u_batch_size])
                normalization = 4*self.u_batch_size # 4 for uniform dist on u and 20 for minibatch on u's

                nabla_w = np.zeros_like(self.w)
                for x1, phi_x1 in zip(x_batch.T, phi_x_batch.T): # loop
                    x1 = x1.reshape(-1,1)
                    phi_x1 = phi_x1.reshape(-1,1)

                    inner_pi_us1 = self.inner_pi_us(u1_batch, x1)
                    inner_pi_us1 = np.real(inner_pi_us1)
                    max_inner_pi_u1 = np.max(inner_pi_us1)
                    pi_us1 = np.exp(inner_pi_us1 - max_inner_pi_u1)
                    Z_x1 = np.sum(pi_us1)

                    pis1 = pi_us1 / (Z_x1*normalization)
                    log_pis1 = np.log(pis1)
                    K_us1 = self.K_us(u1_batch)
                    costs = self.cost(
                        x1 * np.ones([x1.shape[0],self.All_U.shape[1]]),
                        u1_batch
                    )
                    costs_plus_log_pis1 = costs + log_pis1

                    expectationTerm1 = np.sum(pis1.reshape(-1,1) * (costs_plus_log_pis1.reshape(-1,1) + (self.w.T @ K_us1 @ phi_x1).reshape(u1_batch.shape[1],1)))

                    inner_pi_us2 = self.inner_pi_us(u2_batch, x1)
                    inner_pi_us2 = np.real(inner_pi_us2)
                    max_inner_pi_u2 = np.max(inner_pi_us2)
                    pi_us2 = np.exp(inner_pi_us2 - max_inner_pi_u2)
                    Z_x2 = np.sum(pi_us2)

                    pis2 = pi_us2 / (Z_x2 *normalization)
                    #log_pis2 = np.log(pis2)
                    K_us2 = self.K_us(u2_batch)
                    costs = self.cost(
                        x1 * np.ones([x1.shape[0],u2_batch.shape[1]]),
                        u2_batch
                    )
                    expectationTerm2 = np.einsum('i, ijk -> jk', pis2, K_us2 @ phi_x1)

                    # Equation 13/14 in writeup
                    #! Can we just replace first term in multiplication with call to BE method?
                    nabla_w += ((self.w.T @ phi_x1 - expectationTerm1) * (phi_x1 - expectationTerm2)) / batch_size

                gradientNorm = l2_norm(nabla_w, np.zeros_like(nabla_w))
                gradientNorms.append(gradientNorm)

                # Update weights
                assert self.w.shape == nabla_w.shape
                self.w = self.w - (self.learning_rate * nabla_w)
                # print("Current weights:", self.w)

                # Recompute Bellman error
                BE = self.bellmanError()[0,0]
                bellmanErrors.append(BE)
                print("Current Bellman error:", BE)

            return bellmanErrors, gradientNorms
            # while BE > self.epsilon:
            #     # These are col vectors
            #     u1 = np.random.uniform(self.u_lower, self.u_upper)
            #     u2 = np.random.uniform(self.u_lower, self.u_upper)
            #     x1 = self.X[:, np.random.choice(np.arange(self.X.shape[1]))].reshape(-1,1)
            #     phi_x1 = self.phi(x1)
            #     K_u1 = self.K_u(u1)
            #     K_u2 = self.K_u(u2)

            #     # Equation 13/14 in writeup
            #     nabla_w = (self.w @ phi_x1 - ((self.pi_u(u1, x1) / rho(u1, a=0, b=2)) * (self.cost(x1, u1) + np.log(self.pi_u(u1, x1)) + self.w @ K_u1 @ phi_x1))) \
            #                 * (phi_x1 - (self.pi_u(u2, x1) / rho(u2, a=0, b=2)) * K_u2 @ phi_x1)

            #     # Update weights
            #     self.w = self.w - (self.learning_rate * nabla_w)

            #     # Recompute Bellman error
            #     BE = self.bellmanError()
            #     print("Current Bellman error:", BE)

    def Q_pi_t(self, x, u):
        return self.cost(x, u) + self.w @ self.K_u(u)

    def algorithm3(self):
        ''' Policy iteration
        TODO: Include regularization term in PI algo
         '''

        # These are col vectors
        u1 = self.U[:, np.random.choice(np.arange(self.U.shape[1]))].reshape(-1,1) # sample from rho --unif(-2,2) for example
        u2 = self.U[:, np.random.choice(np.arange(self.U.shape[1]))].reshape(-1,1) # sample from rho --unif(-2,2) for example
        x1 = self.X[:, np.random.choice(np.arange(self.X.shape[1]))].reshape(-1,1)

        # get pi_t
        t = 0
        pi_t = [lambda u,x: np.exp(self.inner_pi_u(u,x)) * rho(u)] # pi_t[0] == pi_0
        w_t = [self.w]
        # get w from SGD
        phi_x1 = self.phi(x1)
        for t in range(1, 1000): #? while something > self.epsilon?
            #? keep log in the nabla_w calculation?
            nabla_w = (
                self.w @ phi_x1 - (
                    ( np.exp(self.inner_pi_u(u1, x1)) / rho(u1, a=-2, b=2) ) \
                    * ( self.cost(x1, u1) \
                    + self.w @ self.K_u(u1) @ phi_x1 )
                )
            ) * (
                phi_x1 - ( np.exp(self.inner_pi_u(u2, x1)) / rho(u2, a=-2, b=2) ) \
                * self.K_u(u2) @ phi_x1
            )
            # get w^hat
            w_t.append(self.w - (self.learning_rate * nabla_w))
            self.w = w_t[t]
            # update pi with softmax
            pi_u = lambda u,x: np.exp((-self.learning_rate * (self.cost(x, u) + w_t[t] @ self.K_u(u) @ self.phi(x)))[0])
            pi_t.append(lambda u,x: pi_t[t-1](u) * pi_u(u,x))
            print(f"end loop {t}")

        return pi_t[-1]