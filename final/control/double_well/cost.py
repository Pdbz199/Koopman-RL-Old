# Imports
import numpy as np

from dynamics import state_dim

# Define cost/reward
Q = np.eye(state_dim)
R = 0.001

reference_point = np.array([
    [0],
    [0]
])

def cost(x, u):
    """
        Assuming that data matrices are passed in for X and U. Columns vectors are snapshots.
    """

    # x.T Q x + u.T R u

    _x = x - reference_point

    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    return mat.T

def reward(x, u):
    return -cost(x, u)