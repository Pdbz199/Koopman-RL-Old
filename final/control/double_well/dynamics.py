# Imports
import numpy as np

from scipy.integrate import solve_ivp

# Variables
state_dim = 2
action_dim = 1

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

state_range = 1.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

action_range = 75.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

# Policy that only returns 0
def zero_policy(x=None):
    return np.zeros(action_column_shape)

def random_policy(x=None):
    return np.random.choice(all_actions, size=action_column_shape)

# Dynamics
dt = 0.01

def continuous_f(action=None):
    """
        True, continuous dynamics of the system.

        INPUTS:
            action - Action vector. If left as None, then random policy is used.
    """

    def f_u(t, input):
        """
            INPUTS:
                input - State vector.
                t - Timestep.
        """

        x, y = input
        
        u = action
        if u is None:
            u = zero_policy()

        b_x = np.array([
            [4*x - 4*(x**3)],
            [-2*y]
        ]) + u
        sigma_x = np.array([
            [0.7, x],
            [0, 0.5]
        ])

        column_output = b_x + sigma_x * np.random.randn(2,1)
        x_dot = column_output[0,0]
        y_dot = column_output[1,0]

        return [ x_dot, y_dot ]

    return f_u

def f(state, action):
    """
        True, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.

        INPUTS:
            state - State column vector.
            action - Action column vector.

        OUTPUTS:
            State column vector pushed forward in time.
    """

    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[0, dt], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

# Compute continuous A and B for LQR policy
continuous_A = np.array([
    [-8, 0],
    [0, -2]
])
continuous_B = np.array([
    [1],
    [1]
])

W, V = np.linalg.eig(continuous_A)

print(f"Eigenvalues of continuous A:\n{W}\n")
print(f"Eigenvectors of continuous A:\n{V}\n")