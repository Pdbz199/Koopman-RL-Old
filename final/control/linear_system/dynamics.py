# Imports
import numpy as np

from scipy.special import comb

# Variables
system_name = "linear_system"

state_dim = 3
action_dim = 1

state_column_shape = (state_dim, 1)
action_column_shape = (action_dim, 1)

state_range = 25.0
# state_range = 5.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

# action_range = 75.0
action_range = 10.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

# step_size = 0.1
# step_size = 0.5
step_size = 0.25
# step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

state_order = 2
action_order = 2

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( state_order+state_dim, state_order ) )

def get_random_initial_conditions(num_samples=1):
    return np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, num_samples]
    ).T

# Dynamics
max_eigen_factor = np.random.uniform(0.7, 1)
print(f"max eigen factor: {max_eigen_factor}")
Z = np.random.rand(state_dim, state_dim)
_, sigma, _ = np.linalg.svd(Z)
Z = Z * np.sqrt(max_eigen_factor) / np.max(sigma)
A = Z.T @ Z
W, _ = np.linalg.eig(A)
max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

print(f"A:\n{A}")
print(f"A's max absolute real eigenvalue: {max_abs_real_eigen_val}")

B = np.ones([state_dim, action_dim])

def f(x, u):
    """
        True dynamics of linear system.

        INPUTS:
            x - State as a column vector
            u - Action as a column vector

        OUTPUTS:
            x' - Next state as a column vector
    """

    return A @ x + B @ u

# Controllers
def zero_policy(x=None):
    return np.zeros(action_column_shape)

def random_policy(x=None):
    return np.random.choice(all_actions[0], size=action_column_shape)
