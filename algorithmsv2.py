import mpmath as mp
import numpy as np
import scipy.integrate as integrate

# you can pass functions as parameters to jitted functions IF AND ONLY IF they're also jitted

def rho(u, o='unif', a=0, b=1):
    if o == 'unif':
        return 1 / ( b - a )

    if o == 'normal':
        return np.exp( -u**2 / 2 ) / ( np.sqrt( 2 * np.pi ) )

def K_u(K, psi_u):
    return np.einsum('ijz,z->ij', K, psi_u)

class algos:
    def __init__(self, X, U, u_lower, u_upper, phi, psi, K_hat, cost, learning_rate=0.1, epsilon=1):
        self.X = X
        self.U = U # U is a collection of all POSSIBLE actions as row vectors
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.phi = phi
        self.psi = psi
        self.K_hat = K_hat
        self.num_lifted_state_features = K_hat.shape[0]
        self.num_lifted_action_features = K_hat.shape[2]
        self.cost = cost
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.w = np.ones(self.num_lifted_state_features)
    
    # unnormalized optimal policy
    def pi_u(self, u, x):
       pi_u = mp.exp(-1 * (self.cost(x, u) + self.w @ K_u(self.K_hat, self.psi(u)) @ self.phi(x))) # / Z_x
       return pi_u

    def discreteBellmanError(self, x, phi_x):
        """ Equation 3 in writeup """

        total = 0
        for i in range(self.X.shape[1]):
            pi_us = []
            for u in self.U:
                pi_us.append(self.pi_u(u[0], x))
            Z_x = np.sum(pi_us) # Compute Z_x to use later

            expectation_u = 0
            for i,u in enumerate(self.U):
                u = u[0]
                pi = pi_us[i] / Z_x
                expectation_u += ( self.cost(x, u) - mp.log( pi ) - self.w @ K_u(self.K_hat, self.psi(u)) @ phi_x ) * pi
            total += np.power(( self.w @ phi_x - expectation_u ), 2)
        return total

    def continuousBellmanError(self, x, phi_x):
        """ Equation 3 in writeup modified for continuous action """

        total = 0
        for i in range(self.X.shape[1]):
            Z_x = integrate.quad(self.pi_u, self.u_lower, self.u_upper, (x))[0]
            pi = (lambda u: self.pi_u(u, x) / Z_x)
            expectation_u_integrand =  (lambda u: (self.cost(x, u) - mp.log( pi(u) ) - self.w @ K_u(self.K_hat, self.psi(u)) @ phi_x) * pi(u))
            expectation_u = integrate.quad(expectation_u_integrand, self.u_lower, self.u_upper)[0]
            total += np.power(( self.w @ phi_x - expectation_u ), 2)
        return total

    def algorithm2(self):
        """ Bellman error optimization """

        # Sample initial state x1 from sample of snapshots
        x1 = self.X[:, int(np.random.uniform(0, self.X.shape[1]))]
        BE = self.continuousBellmanError(x1, self.phi(x1))

        while BE > self.epsilon:
            # These are row vectors
            u1 = np.random.uniform(-2, 2) #sample from rho --unif(-2,2) for example
            u2 = np.random.uniform(-2, 2) #sample from rho --unif(-2,2) for example
            x1 = self.X[:, int(np.random.uniform(0, self.X.shape[1]))]

            phi_x1 = self.phi(x1)
            # Equation 4/5 in writeup
            nabla_w = (
                self.w @ phi_x1 - (
                    ( self.pi_u(u1, x1) / rho(u1, a=-2, b=2) ) \
                  * ( self.cost(x1, u1) + mp.log( self.pi_u(u1, x1) ) \
                  + self.w @ K_u(self.K_hat, self.psi(u1)) @ phi_x1 )
                )
            ) * (
                phi_x1 - ( self.pi_u(u2, x1) / rho(u2, a=-2, b=2) ) \
              * K_u(self.K_hat, self.psi(u2)) @ phi_x1
            )

            # Update weights
            self.w = self.w - (self.learning_rate * nabla_w)

            BE = self.continuousBellmanError(x1, phi_x1)
            print(BE)