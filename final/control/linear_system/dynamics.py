# Imports
import numpy as np

# Variables
state_dim = 3
action_dim = 1

# state_range = 25.0
state_range = 5.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

# action_range = 75.0
action_range = 10.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

step_size = 0.1
# step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

state_order = 2
action_order = 2

# Dynamics
max_eigen_factor = np.random.uniform(0.7, 1)
print(f"max eigen factor: {max_eigen_factor}")
Z = np.random.rand(state_dim, state_dim)
_, sigma, _ = np.linalg.svd(Z)
Z = Z * np.sqrt(max_eigen_factor) / np.max(sigma)
A = Z.T @ Z
W, _ = np.linalg.eig(A)
max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

print(f"A: {A}")
print(f"A's max absolute real eigenvalue: {max_abs_real_eigen_val}")

B = np.ones([state_dim, action_dim])

def f(x, u):
    return A @ x + B @ u