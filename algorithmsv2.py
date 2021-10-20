import mpmath as mp
import numpy as np
import scipy.integrate as integrate
import time
import auxiliaries as aux

def rho(u, o='unif', a=0, b=1):
    if o == 'unif':
        return 1 / ( b - a )

    if o == 'normal':
        return np.exp( -u**2 / 2 ) / ( np.sqrt( 2 * np.pi ) )

def K_u(K, psi_u):
    ''' Pick out Koopman operator given a particular action '''

    # if psi_u.shape == 2:
    #     psi_u = psi_u[:,0]
    return np.einsum('ijz,z->ij', K, psi_u)

class algos:
    def __init__(self, X, U, u_lower, u_upper, phi, psi, K_hat, cost, bellmanErrorType=0, learning_rate=0.1, epsilon=1, weightRegularizationBool = 1, weightRegLambda = 1e-2):
        self.X = X # Collection of observations
        self.U = U # U is a collection of all POSSIBLE actions as row vectors
        self.u_lower = u_lower # lower bound on actions
        self.u_upper = u_upper # upper bound on actions
        self.phi = phi # Dictionary function for X
        self.psi = psi # Dictionary function for U
        self.K_hat = K_hat # Estimated Koopman Tensor
        self.cost = cost # Cost function to optimize
        self.bellmanError = self.discreteBellmanError if bellmanErrorType == 0 else self.continuousBellmanError
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.w = np.ones(K_hat.shape[0]) # Default weights of 1s
        self.weightRegularization = weightRegularizationBool #Bool for including weight regularization in Bellman loss functions
        self.weightRegLambda = weightRegLambda
        
    def pi_u(self, u, x):
        ''' Unnormalized optimal policy '''

        K_u_const = K_u(self.K_hat, self.psi(u)[:,0])
        pi_u = mp.exp((-self.learning_rate * (self.cost(x, u) + self.w @ K_u_const @ self.phi(x)))[0])
        return pi_u

    def discreteBellmanError(self):
        ''' Equation 3 in writeup with weight regularization added to help gradient explosion in Bellman algos '''

        total = 0
        for i in range(self.X.shape[1]):
            x = self.X[:,i].reshape(-1,1)
            phi_x = self.phi(x)[:,0]

            pi_us = []
            for u in self.U:
                u = u.reshape(-1,1)
                pi_us.append(self.pi_u(u, x))
            Z_x = np.sum(pi_us) # Compute Z_x to use later

            expectation_u = 0
            for i,u in enumerate(self.U):
                u = u.reshape(-1,1)
                pi = pi_us[i] / Z_x
                K_u_const = K_u(self.K_hat, self.psi(u)[:,0])
                expectation_u += ( self.cost(x, u) - mp.log(pi) - self.w @ K_u_const @ phi_x ) * pi
            total += np.power(( self.w @ phi_x - expectation_u ), 2)

        # add weight regularization term to help with gradient explosion issues
        # if(self.weightRegularization == 1):
        #     total += self.weightRegLambda*(aux.l2_norm(self.w)**2)

        return total

    def continuousBellmanError(self):
        ''' Equation 3 in writeup modified for continuous action weight regularization added to help gradient explosion in Bellman algos '''

        pi = (lambda u, x, Z_x: self.pi_u(u, x) / Z_x)
        def expectation_u_integrand(u, x, phi_x, Z_x):
            K_u_const = K_u(self.K_hat, self.psi(np.array([[u]]))[:,0])
            pi_u_const = pi(np.array([[u]]), x, Z_x)
            return (self.cost(x, u) - mp.log(pi_u_const) - self.w @ K_u_const @ phi_x) * pi_u_const

        total = 0
        for i in range(self.X.shape[1]):
            x = self.X[:,i].reshape(-1,1)
            phi_x = self.phi(x)[:,0]

            Z_x = integrate.quad(self.pi_u, self.u_lower, self.u_upper, (x))[0]
            expectation_u = integrate.quad(expectation_u_integrand, self.u_lower, self.u_upper, (x, phi_x, Z_x))[0]

            total += np.power(( self.w @ phi_x - expectation_u ), 2)

        #add weight regularization term to help with gradient explosion issues
        # if(self.weightRegularization == 1):
        #     total += self.weightRegLambda*(aux.l2_norm(self.w)**2)

        return total

    def algorithm2(self):
        ''' Bellman error optimization '''

        BE = self.bellmanError()

        while BE > self.epsilon:
            print(BE)

            # These are col vectors
            u1 = np.array([[np.random.uniform(self.u_lower, self.u_upper)]]) # sample from rho --unif(u_lower,u_upper) for example
            u2 = np.array([[np.random.uniform(self.u_lower, self.u_upper)]]) # sample from rho --unif(u_lower,u_upper) for example
            x1 = self.X[:, int(np.random.uniform(0, self.X.shape[1]))].reshape(-1,1)

            phi_x1 = self.phi(x1)[:,0]
            # Equation 4/5 in writeup
            nabla_w = (
                self.w @ phi_x1 - (
                    ( self.pi_u(u1, x1) / rho(u1, a=-2, b=2) ) \
                  * ( self.cost(x1, u1) + mp.log( self.pi_u(u1, x1) ) \
                  + self.w @ K_u(self.K_hat, self.psi(u1)[:,0]) @ phi_x1 )
                )
            ) * (
                phi_x1 - ( self.pi_u(u2, x1) / rho(u2, a=-2, b=2) ) \
              * K_u(self.K_hat, self.psi(u2)[:,0]) @ phi_x1
            )

            # Update weights
            self.w = self.w - (self.learning_rate * nabla_w) - 2*self.weightRegLambda*self.w

            BE = self.bellmanError()

    def Q_pi_t(self, x, u):
        return self.cost(x, u) + self.w @ K_u(self.K_hat, psi(u))

    def algorithm3(self):
        ''' Policy iteration
        TODO: Include regularization term in PI algo
         '''

        # These are col vectors
        u1 = np.array([[np.random.uniform(-2, 2)]]) # sample from rho --unif(-2,2) for example
        u2 = np.array([[np.random.uniform(-2, 2)]]) # sample from rho --unif(-2,2) for example
        x1 = self.X[:, int(np.random.uniform(0, self.X.shape[1]))].reshape(-1,1)

        # get pi_t
        t = 0
        pi_t = [lambda u,x: self.pi_u(u,x) * rho(u)] # pi_t[0] == pi_0
        w_t = [self.w]
        # get w from SGD
        phi_x1 = self.phi(x1)[:,0]
        for t in range(1, 1000): #? while something > self.epsilon?
            #? keep log in the nabla_w calculation?
            nabla_w = (
                self.w @ phi_x1 - (
                    ( self.pi_u(u1, x1) / rho(u1, a=-2, b=2) ) \
                    * ( self.cost(x1, u1) \
                    + self.w @ K_u(self.K_hat, self.psi(u1)[:,0]) @ phi_x1 )
                )
            ) * (
                phi_x1 - ( self.pi_u(u2, x1) / rho(u2, a=-2, b=2) ) \
                * K_u(self.K_hat, self.psi(u2)[:,0]) @ phi_x1
            )
            # get w^hat
            w_t.append(self.w - (self.learning_rate * nabla_w))
            self.w = w_t[t]
            # update pi with softmax
            pi_u = lambda u,x: mp.exp((-self.learning_rate * (self.cost(x, u) + w_t[t] @ K_u(self.K_hat, self.psi(u)[:,0]) @ self.phi(x)))[0])
            pi_t.append(lambda u,x: pi_t[t-1](u) * pi_u(u,x))
            print(f"end loop {t}")

        return pi_t[-1]