import math
import numpy as np
import tensorflow as tf

def rho(u, o='unif', a=0, b=1):
    if o == 'unif':
        return tf.divide(1, tf.subtract(b, a))

    if o == 'normal':
        neg_u_squared = tf.math.square(-u,2)
        neg_u_squared_over_2 = tf.divide(neg_u_squared, 2)
        two_pi = tf.multiply(2, tf.constant(math.pi))
        sqrt_two_pi = tf.math.sqrt(two_pi)
        return tf.divide(tf.math.exp(neg_u_squared_over_2), sqrt_two_pi)

def K_u(K, psi_u):
    ''' Pick out Koopman operator given a particular action '''
    return tf.cast(tf.einsum('ijz,z->ij', K, psi_u[:,0]), tf.float32)

class Algorithms:
    def __init__(self, N, X, U, phi, psi, K_hat, cost):
        self.N = N # Number of data points (X.shape[1])
        self.X = X # Sampled data points
        self.U = U # Contains all possible actions (if discrete)
        self.u_bounds = tf.stack([tf.math.reduce_min(U), tf.math.reduce_max(U)])
        self.num_unique_actions = self.U.shape[1]
        self.phi = phi
        self.psi = psi
        self.K_hat = K_hat # Estimated Koopman Tensor
        self.cost = cost
        self.w = tf.Variable(tf.ones(K_hat.shape[0]), name='w') # Default weights of 1s

        self.optimizer = tf.keras.optimizers.SGD()

        self.pi = lambda u,x: tf.divide(1, self.num_unique_actions)

    def computeSoftMaxOptimalPolicy(self):
        ''' Equation 11 in writeup '''
        def pi(u,x):
            phi_x = tf.cast(tf.stack(self.phi(x)), tf.float32)
            psi_u = tf.cast(tf.stack(self.psi(u)), tf.float32)

            def pi_u(u):
                return tf.cast(self.pi_u(u, x), tf.float32) #? tf.stack()
            pi_us = tf.map_fn(fn=pi_u, elems=self.U, dtype=tf.float32)
            Z_x = tf.math.reduce_sum(pi_us) # Compute Z_x to use later

            phi_x_prime = tf.tensordot(K_u(self.K_hat, psi_u), phi_x)
            inner = tf.add(tf.math.log(self.cost(x,u)), tf.tensordot(self.w, phi_x_prime))
            numerator = tf.math.exp(-inner)

            return tf.divide(numerator, Z_x)

        self.pi = pi

    def discreteBellmanError(self, x):
        ''' Equation 12 in writeup '''

        def computeError(i):
            phi_x = tf.cast(tf.stack(self.phi(x)), tf.float32)

            random_index = tf.random.shuffle(tf.range(self.num_unique_actions))[0]
            u1 = self.U[:,random_index]
            u1 = tf.reshape(u1, [tf.shape(u1)[0],1])
            random_index = tf.random.shuffle(tf.range(self.num_unique_actions))[0]
            u2 = self.U[:,random_index]
            u2 = tf.reshape(u2, [tf.shape(u2)[0],1])
            psi_u1 = tf.cast(tf.stack(self.psi(u1)), tf.float32)
            psi_u2 = tf.cast(tf.stack(self.psi(u2)), tf.float32)
            # First term of value fn expressed in terms of dictionary
            inner_part_1 = tf.tensordot(self.w, phi_x, axes=1)
            # Computing terms in RHS of Bellman eqn
            cost_plus_log_pi = tf.cast(
                tf.add(self.cost(x,u1), tf.math.log(self.pi(u1,x))),
                tf.float32
            )
            phi_x_prime = tf.tensordot(K_u(self.K_hat, psi_u2), phi_x, axes=1)
            weighted_phi_x_prime = tf.tensordot(self.w, phi_x_prime, axes=1)

            inner_part_2 = tf.add(cost_plus_log_pi, weighted_phi_x_prime)
            importanceWeight = tf.cast(
                tf.multiply(self.pi(u1, x), self.num_unique_actions),
                tf.float32
            )
            inner_part_2 = tf.multiply(importanceWeight, inner_part_2)
            
            inner_difference = tf.subtract(inner_part_1, inner_part_2)
            squared_inner = tf.math.square(inner_difference)
            return squared_inner

        results = tf.map_fn(fn=computeError, elems=tf.range(1), dtype=tf.float32) #self.N

        return tf.math.reduce_sum(results)

    def algorithm2(self):
        random_index = tf.random.shuffle(tf.range(self.N))[0]
        x1 = self.X[:,random_index]
        x1 = tf.reshape(x1, [tf.shape(x1)[0],1])
        bellmanError = self.discreteBellmanError(x1)

        while bellmanError >= 1e-4:
            # Update weights
            for _ in range(100):
                random_index = tf.random.shuffle(tf.range(self.N))[0]
                x1 = self.X[:,random_index]
                x1 = tf.reshape(x1, [tf.shape(x1)[0],1])
                with tf.GradientTape() as tape:
                    loss = self.discreteBellmanError(x1)
                grads = tape.gradient(loss, [self.w])
                self.optimizer.apply_gradients(zip(grads, [self.w]))

            # Recompute bellman error
            random_index = tf.random.shuffle(tf.range(self.N))[0]
            x1 = self.X[:,random_index]
            x1 = tf.reshape(x1, [tf.shape(x1)[0],1])
            bellmanError = self.discreteBellmanError(x1)

            print(bellmanError)