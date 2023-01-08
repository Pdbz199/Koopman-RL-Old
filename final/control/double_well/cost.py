# Imports
import numpy as np

# from dynamics import state_dim #, action_dim
state_dim = 2

# Define cost/reward
Q = np.eye(state_dim)
R = 0.001
# R = np.eye(action_dim) * R

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
    # mat = np.vstack(np.diag(_x.T @ Q @ _x)) + mat = np.vstack(np.diag(u.T @ R @ u))
    return mat.T

def reward(x, u):
    return -cost(x, u)