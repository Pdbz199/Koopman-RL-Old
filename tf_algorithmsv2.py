import math
import numpy as np
import tensorflow as tf
import time

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
        self.w = tf.Variable(tf.ones(K_hat.shape[0]), name='weights') # Default weights of 1s
        # self.pi = lambda u, x: tf.divide(1, self.num_unique_actions)

        self.optimizer = tf.keras.optimizers.SGD()

    def pi(self, u, x, w):
        phi_x = tf.cast(tf.stack(self.phi(x)), tf.float32)
        psi_u = tf.cast(tf.stack(self.psi(u)), tf.float32)

        @tf.autograph.experimental.do_not_convert
        def compute_numerator(i):
            u = self.U[:,i]
            u = tf.reshape(u, [tf.shape(u)[0],1])
            psi_u = tf.cast(tf.stack(self.psi(u)), tf.float32)

            phi_x_prime = tf.tensordot(K_u(self.K_hat, psi_u), phi_x, axes=1)
            weighted_phi_x_prime = tf.tensordot(w, phi_x_prime, axes=1)
            inner = tf.add(self.cost(x,u), weighted_phi_x_prime)
            return tf.math.exp(-inner)
        Z_x = tf.math.reduce_sum(tf.map_fn(fn=compute_numerator, elems=tf.range(self.U.shape[1]), dtype=tf.float32))

        phi_x_prime = tf.tensordot(K_u(self.K_hat, psi_u), phi_x, axes=1)
        weighted_phi_x_prime = tf.tensordot(w, phi_x_prime, axes=1)
        inner = tf.add(self.cost(x,u), weighted_phi_x_prime)
        numerator = tf.math.exp(-inner)

        pi_value = tf.divide(numerator, Z_x)

        return pi_value

    def discreteBellmanError(self, x, w):
        ''' Equation 12 in writeup '''

        @tf.autograph.experimental.do_not_convert
        def computeError(i):
            phi_x = tf.cast(tf.stack(self.phi(x)), tf.float32)

            # Sample u1 and u2, get psi_u1 and psi_u2
            random_index = tf.random.shuffle(tf.range(self.num_unique_actions))[0]
            u1 = self.U[:,random_index]
            u1 = tf.reshape(u1, [tf.shape(u1)[0],1])
            random_index = tf.random.shuffle(tf.range(self.num_unique_actions))[0]
            u2 = self.U[:,random_index]
            u2 = tf.reshape(u2, [tf.shape(u2)[0],1])
            psi_u1 = tf.cast(tf.stack(self.psi(u1)), tf.float32)
            psi_u2 = tf.cast(tf.stack(self.psi(u2)), tf.float32)

            # First term of value fn expressed in terms of dictionary
            inner_part_1 = tf.tensordot(w, phi_x, axes=1)
            # Computing terms in RHS of Bellman eqn
            cost_plus_log_pi = tf.cast(
                tf.add(self.cost(x, u1), tf.math.log(self.pi(u1, x, w))),
                tf.float32
            )
            phi_x_prime = tf.tensordot(K_u(self.K_hat, psi_u2), phi_x, axes=1)
            weighted_phi_x_prime = tf.tensordot(w, phi_x_prime, axes=1)

            inner_part_2 = tf.add(cost_plus_log_pi, weighted_phi_x_prime)
            importanceWeight = tf.cast(
                tf.multiply(self.pi(u1, x, w), self.num_unique_actions),
                tf.float32
            )
            inner_part_2 = tf.multiply(importanceWeight, inner_part_2)
            
            inner_difference = tf.subtract(inner_part_1, inner_part_2)
            squared_inner = tf.math.square(inner_difference)
            return squared_inner

        results = tf.map_fn(fn=computeError, elems=tf.range(self.N/100), dtype=tf.float32)

        return tf.math.reduce_sum(results)

    def algorithm2(self):
        x1 = self.X[:, tf.random.shuffle(tf.range(self.N))[0]]
        x1 = tf.reshape(x1, [tf.shape(x1)[0],1])
        bellmanError = self.discreteBellmanError(x1, self.w)
        print("Initial bellman error:", bellmanError)

        while bellmanError >= 1e-4:
            # Update weights
            w = self.w
            # for _ in range(1): # self.N
            x1 = self.X[:, tf.random.shuffle(tf.range(self.N))[0]]
            x1 = tf.reshape(x1, [tf.shape(x1)[0],1])
            with tf.GradientTape() as tape:
                loss = self.discreteBellmanError(x1, w)
            grads = tape.gradient(loss, [self.w])
            self.optimizer.apply_gradients(zip(grads, [self.w]))

            # Recompute bellman error
            x1 = self.X[:, tf.random.shuffle(tf.range(self.N))[0]]
            x1 = tf.reshape(x1, [tf.shape(x1)[0],1])
            bellmanError = self.discreteBellmanError(x1, self.w)

            print("Current bellman error:", bellmanError)