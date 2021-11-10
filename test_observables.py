import observables
import tensorflow as tf
import tf_observables

x = tf.stack([[1],[5]])

order = 3
phi = observables.monomials(order)
tf_phi = tf_observables.monomials(order)

phi(x)
tf_phi(x)