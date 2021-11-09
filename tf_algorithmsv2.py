import tensorflow as tf

class Algorithms:
    def __init__(self, X, All_U, phi, psi, K_hat, cost, epsilon=1e-4):
        self.X = X # Sampled data points
        self.N = X.shape[1] # Number of data points in dataset
        self.All_U = All_U # Contains all possible actions (if discrete)
        self.u_bounds = tf.stack([tf.math.reduce_min(All_U), tf.math.reduce_max(All_U)])
        self.num_unique_actions = All_U.shape[1]
        self.phi = phi # Dictionary function for X
        self.psi = psi # Dictionary function for U
        self.K_hat = K_hat # Estimated Koopman Tensor
        self.cost = cost # Cost function to optimize
        self.epsilon = epsilon
        self.w = tf.Variable(tf.ones([K_hat.shape[0],1]), name='weights') # Default weights of 1s

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def K_u(self, u):
        ''' Pick out Koopman operator given a particular action '''
        psi_u = self.psi(u)[:,0]
        return tf.cast(tf.einsum('ijz,z->ij', self.K_hat, psi_u), tf.float32)

    def inner_pi_u(self, u, x):
        phi_x_prime = tf.linalg.matmul(self.K_u(u), self.phi(x)) # Column vector
        weighted_phi_x_prime = tf.linalg.matmul(tf.transpose(self.w), phi_x_prime) # Shape: (1,1)
        inner = tf.add(self.cost(x, u), weighted_phi_x_prime)

        return -inner

    def pi_u(self, u, x):
        return tf.math.exp(self.inner_pi_u(u, x))

    # def pi(self, u, x):
    #     @tf.autograph.experimental.do_not_convert
    #     def compute_numerator(i):
    #         u = self.U[:,i]
    #         u = tf.reshape(u, [tf.shape(u)[0],1])
    #         return self.pi_u(u, x)

    #     Z_x = tf.math.reduce_sum(tf.map_fn(fn=compute_numerator, elems=tf.range(self.U.shape[1]), dtype=tf.float32))

    #     numerator = self.pi_u(u, x)

    #     pi_value = tf.divide(numerator, Z_x)

    #     return pi_value

    def discreteBellmanError(self):
        ''' Equation 12 in writeup '''

        @tf.autograph.experimental.do_not_convert
        def computeError(i):
            x = self.X[:,i].reshape(-1,1)
            phi_x = self.phi(x)

            inner_pi_us = []
            for u in self.All_U.T:
                u = u.reshape(-1,1)
                inner_pi_us.append(self.inner_pi_u(u, x))
            # inner_pi_us = np.real(inner_pi_us)
            max_inner_pi_u = tf.math.reduce_max(inner_pi_us)
            pi_us = tf.math.exp(inner_pi_us - max_inner_pi_u)
            Z_x = tf.math.reduce_sum(pi_us)

            expectation_u = 0
            pi_sum = 0
            
            # pis = pi_us / Z_x

            for i,u in enumerate(self.All_U.T):
                u = u.reshape(-1,1)
                pi = pi_us[i] / Z_x
                pi_sum += pi
                expectation_u += (self.cost(x, u) + tf.math.log(pi) + tf.transpose(self.w) @ self.K_u(u) @ phi_x) * pi
            error = tf.math.pow((tf.transpose(self.w) @ phi_x - expectation_u), 2)

            return error

        totals = tf.map_fn(fn=computeError, elems=tf.range(self.N), dtype=tf.float32)

        return tf.math.reduce_sum(totals)

    def algorithm2(self):
        # Compute initial Bellman error (before any updates)
        bellmanError = self.discreteBellmanError()
        print("Initial bellman error:", bellmanError)

        # Loop until convergence (while weights are not good enough)
        while bellmanError >= self.epsilon:
            # Run gradient descent with TensorFlow
            with tf.GradientTape() as tape: # Tell TensorFlow to remember the computation within for gradient computation
                loss = self.discreteBellmanError() # Compute Bellman error with current weights
            grads = tape.gradient(loss, [self.w]) # Compute gradient with respect to weights
            self.optimizer.apply_gradients(zip(grads, [self.w])) # Descend via back propogation (update weights)

            # Recompute bellman error with new weights
            bellmanError = self.discreteBellmanError()
            print("Current bellman error:", bellmanError)