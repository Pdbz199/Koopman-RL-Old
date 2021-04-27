#%%
import observables
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

B = tf.Variable(tf.ones((3,2)))

# Psi_X = dictionaries applied to x
# nabla_X.T @ (B.T @ Psi_X)
# monomials(2) = 1, x_1, x_2, x_1**2, x_1*x_2, x_2**2

#%%
psi = observables.monomials(2)
Psi_X = psi(tf.constant([[2,2]]))

# %%
with tf.GradientTape(persistent=True) as tape:
    y = tf.transpose(B) @ Psi_X

# Not really sure about correctness...
grad = tape.gradient(y, B)

# %%
