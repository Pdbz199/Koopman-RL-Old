# Imports
import numpy as np

from scipy.integrate import solve_ivp

# Variables
state_dim = 3
action_dim = 1

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

state_range = 25.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

action_range = 10.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

state_order = 2
action_order = 2

# Default basic policies
def zero_policy(x=None):
    return np.zeros(action_column_shape)

def random_policy(x=None):
    return np.random.choice(all_actions[0], size=action_column_shape)

# Dynamics
omega = 1.0
mu = 0.1
A = -0.1
lamb = 1

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

        x, y, z = input

        x_dot = mu*x - omega*y + A*x*z
        y_dot = omega*x + mu*y + A*y*z
        z_dot = -lamb * ( z - np.power(x, 2) - np.power(y, 2) )

        u = action
        if u is None:
            u = zero_policy()

        return [ x_dot, y_dot + u, z_dot ]

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
x_bar = 0
y_bar = 0
z_bar = 0

continuous_A = np.array([
    [mu + A * z_bar, -omega, A * x_bar],
    [omega, mu + A * z_bar, A * y_bar],
    [2 * lamb * x_bar, 2 * lamb * y_bar, -lamb]
])
continuous_B = np.array([
    [0],
    [1],
    [0]
])

W, V = np.linalg.eig(continuous_A)

print(f"Eigenvalues of continuous A:\n{W}\n")
print(f"Eigenvectors of continuous A:\n{V}\n")