# Imports
import numpy as np

from scipy.integrate import solve_ivp
from scipy.special import comb

# Variables
system_name = "fluid_flow"

state_dim = 3
action_dim = 1

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

# state_range = 25.0
state_range = 1.0
state_minimums = np.array([
    [-state_range],
    [-state_range],
    [0.0]
])
state_maximums = np.array([
    [state_range],
    [state_range],
    [1.0]
])

action_range = 10.0
action_minimums = np.array([
    [-action_range]
])
action_maximums = np.array([
    [action_range]
])

state_order = 2
action_order = 2

# step_size = 0.5
step_size = 0.025
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

# state_order = 2
# action_order = 2
state_order = 3
action_order = 3

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( state_order+state_dim, state_order ) )

def get_random_initial_conditions(num_samples=1, on_limit_cycle=True):
    if on_limit_cycle:
        initial_states = np.zeros([num_samples, state_dim])
        for episode in range(num_samples):
            x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
            u = np.array([[0]])
            soln = solve_ivp(fun=continuous_f(u), t_span=[0, 30.0], y0=x[:, 0], method='RK45')
            initial_states[episode] = soln.y[:,-1]
        return initial_states

    return np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, num_samples]
    ).T

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

        # return [ x_dot, y_dot + u, z_dot ]
        return [ x_dot, y_dot + u[0], z_dot ]

    return f_u

# def f(state, action):
#     """
#         True, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.

#         INPUTS:
#             state - State column vector.
#             action - Action column vector.

#         OUTPUTS:
#             State column vector pushed forward in time.
#     """

#     # soln = solve_ivp(fun=continuous_f(action[:, 0]), t_span=[0, dt], y0=state[:, 0], method='RK45')
#     soln = np.array(continuous_f(action[:, 0])(0, state[:, 0])) * dt

#     # return np.vstack(soln.y[:, -1])
#     return np.vstack(soln)

def f(state, action):
    """
        True, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.

        INPUTS:
            state - State column vector.
            action - Action column vector.

        OUTPUTS:
            State column vector pushed forward in time.
    """

    x = state[0, 0]
    y = state[1, 0]
    z = state[2, 0]

    return state + (np.array([
        [mu*x - omega*y + A*x*z],
        [omega*x + mu*y + A*y*z + action[0, 0]],
        [-lamb * ( z - np.power(x, 2) - np.power(y, 2) )]
    ]) * dt)

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
