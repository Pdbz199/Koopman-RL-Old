import auxiliaries as aux
import numpy as np
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
    def __init__(self, X, All_U, u_lower, u_upper, phi, psi, K_hat, cost, bellmanErrorType=0, learning_rate=1e-2, epsilon=1, weightRegularizationBool = 1, weightRegLambda = 1e-2):
        self.X = X # Collection of observations
        self.All_U = All_U # U is a collection of all POSSIBLE actions as row vectors
        self.u_lower = u_lower # lower bound on actions
        self.u_upper = u_upper # upper bound on actions
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
        ''' Pick out Koopman operator given a particular action '''
        return np.einsum('ijz,zk->ij', self.K_hat, self.psi(u))

    def inner_pi_u(self, u, x):
        inner_pi_u = (-(self.cost(x, u) + self.w.T @ self.K_u(u) @ self.phi(x)))[0]
        return inner_pi_u

    def pi_u(self, u, x):
        inner = self.inner_pi_u(u, x)
        return np.exp(inner)

    def discreteBellmanError(self):
        ''' Equation 12 in writeup '''

        # TODO: Vectorize
        total = 0
        for i in range(self.X.shape[1]):
            x = self.X[:,i].reshape(-1,1)
            phi_x = self.phi(x)

            inner_pi_us = []
            for u in self.All_U.T:
                u = u.reshape(-1,1)
                inner_pi_us.append(self.inner_pi_u(u, x))
            inner_pi_us = np.real(inner_pi_us)
            max_inner_pi_u = np.max(inner_pi_us)
            pi_us = np.exp(inner_pi_us - max_inner_pi_u)
            Z_x = np.sum(pi_us)

            expectation_u = 0
            pi_sum = 0
            for i,u in enumerate(self.All_U.T):
                u = u.reshape(-1,1)
                pi = pi_us[i] / Z_x
                assert pi >= 0
                pi_sum += pi
                expectation_u += (self.cost(x, u) + np.log(pi) + self.w.T @ self.K_u(u) @ phi_x) * pi
            total += np.power((self.w.T @ phi_x - expectation_u), 2) #/ self.X.shape[1]
            assert np.isclose(pi_sum, 1, rtol=1e-3, atol=1e-4)
        return total

    def continuousBellmanError(self):
        ''' Equation 3 in writeup modified for continuous action weight regularization added to help gradient explosion in Bellman algos '''

        pi = (lambda u, x, Z_x: np.exp(self.inner_pi_u(u, x)) / Z_x)
        def expectation_u_integrand(u, x, phi_x, Z_x):
            pi_u_const = pi(np.array([[u]]), x, Z_x)
            return (self.cost(x, u) - np.log(pi_u_const) - self.w.T @ self.K_u(np.array([[u]])) @ phi_x) * pi_u_const

        total = 0
        for i in range(self.X.shape[1]):
            x = self.X[:,i].reshape(-1,1)
            phi_x = self.phi(x)

            Z_x = integrate.quad(np.exp(self.inner_pi_u), self.u_lower, self.u_upper, (x))[0]
            expectation_u = integrate.quad(expectation_u_integrand, self.u_lower, self.u_upper, (x, phi_x, Z_x))[0]

            total += np.power(( self.w.T @ phi_x - expectation_u ), 2)

        #add weight regularization term to help with gradient explosion issues
        # if self.weightRegularization:
        #     total += self.weightRegLambda*(aux.l2_norm(self.w)**2)

        return total

    def algorithm2(self):
        ''' Bellman error optimization '''
        
        batch_size = 256

        BE = self.bellmanError()[0,0]
        bellmanErrors = [BE]
        gradientNorms = []
        print("Initial Bellman error:", BE)

        if not self.bellmanErrorType: # if discrete BE
            while BE > self.epsilon:
                x_batch = np.empty([self.X.shape[0],batch_size])
                for i in range(batch_size):
                    x = self.X[:, np.random.choice(np.arange(self.X.shape[1]))]
                    x_batch[:,i] = x
                phi_x_batch = self.phi(x_batch)

                nabla_w = np.zeros_like(self.w)
                for x1, phi_x1 in zip(x_batch.T, phi_x_batch.T):
                    x1 = x1.reshape(-1,1)
                    phi_x1 = phi_x1.reshape(-1,1)

                    inner_pi_us = []
                    for u in self.All_U.T:
                        u = u.reshape(-1,1)
                        inner_pi_us.append(self.inner_pi_u(u, x1))
                    inner_pi_us = np.real(inner_pi_us)
                    max_inner_pi_u = np.max(inner_pi_us)
                    pi_us = np.exp(inner_pi_us - max_inner_pi_u)
                    Z_x = np.sum(pi_us)

                    expectationTerm1 = 0
                    expectationTerm2 = 0
                    for i,u in enumerate(self.All_U.T):
                        u = u.reshape(-1,1)
                        K_u = self.K_u(u)
                        expectationTerm1 += (pi_us[i] / Z_x) * (self.cost(x1, u) + np.log(pi_us[i] / Z_x) + self.w.T @ K_u @ phi_x1)
                        expectationTerm2 += (pi_us[i] / Z_x) * K_u @ phi_x1

                    # Equation 13/14 in writeup
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
        else:
            while BE > self.epsilon:
                # These are col vectors
                u1 = self.All_U[:, np.random.choice(np.arange(self.All_U.shape[1]))].reshape(-1,1)
                u2 = self.All_U[:, np.random.choice(np.arange(self.All_U.shape[1]))].reshape(-1,1)
                x1 = self.X[:, np.random.choice(np.arange(self.X.shape[1]))].reshape(-1,1)
                phi_x1 = self.phi(x1)
                K_u1 = self.K_u(u1)
                K_u2 = self.K_u(u2)

                # Equation 13/14 in writeup
                nabla_w = (self.w @ phi_x1 - ((self.pi_u(u1, x1) / rho(u1, a=0, b=2)) * (self.cost(x1, u1) + np.log(self.pi_u(u1, x1)) + self.w @ K_u1 @ phi_x1))) \
                            * (phi_x1 - (self.pi_u(u2, x1) / rho(u2, a=0, b=2)) * K_u2 @ phi_x1)

                # Update weights
                self.w = self.w - (self.learning_rate * nabla_w)

                # Recompute Bellman error
                BE = self.bellmanError()
                print("Current Bellman error:", BE)

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