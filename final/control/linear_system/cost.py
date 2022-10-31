# Imports
import numpy as np

from dynamics import state_dim, action_dim

# Define cost/reward
Q = np.eye(state_dim)
# R = 1
R = np.eye(action_dim)

reference_point = np.array([
    [0.0],
    [0.0],
    [0.0]
])

def cost(x, u):
    """
        Assuming that data matrices are passed in for X and U. Columns vectors are snapshots.
    """

    _x = x - reference_point

    # return _x.T @ Q @ _x + u.T @ R @ u

    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.vstack(np.diag(u.T @ R @ u))
    return mat.T

def reward(x, u):
    return -cost(x, u)