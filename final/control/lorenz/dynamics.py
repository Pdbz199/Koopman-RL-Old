# Imports
import numpy as np

from scipy.integrate import solve_ivp
from scipy.special import comb

# Variables
state_dim = 3
action_dim = 1

state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

phi_column_shape = [phi_dim, 1]

# state_range = 50.0
# state_minimums = np.ones([state_dim,1]) * -state_range
# state_maximums = np.ones([state_dim,1]) * state_range
state_minimums = np.array([
    [-20.0],
    [-50.0],
    [0.0]
])
state_maximums = np.array([
    [20.0],
    [50.0],
    [50.0]
])

def get_random_initial_conditions(num_samples=1):
    return np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, num_samples]
    ).T

# action_range = 75.0
action_range = 500.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

# Default basic policies
def zero_policy(x=None):
    return np.zeros(action_column_shape)

def random_policy(x=None):
    return np.random.choice(all_actions[0], size=action_column_shape)

# Dynamics
sigma = 10
rho = 28
beta = 8/3

dt = 0.01

x_e = np.sqrt( beta * ( rho - 1 ) )
y_e = np.sqrt( beta * ( rho - 1 ) )
z_e = rho - 1

def continuous_f(action=None):
    """
        True, continuous dynamics of the system.

        INPUTS:
            action - Action vector. If left as None, then random policy is used.
    """

    def f_u(t, input):
        """
            INPUTS:
                t - Timestep.
                input - State vector.
        """

        x, y, z = input

        # x = x - x_e
        # y = y - y_e
        # z = z + z_e

        x_dot = sigma * ( y - x )   # sigma*y - sigma*x
        y_dot = ( rho - z ) * x - y # rho*x - x*z - y
        z_dot = x * y - beta * z    # x*y - beta*z

        u = action
        if u is None:
            u = zero_policy()

        return [ x_dot + u, y_dot, z_dot ]

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

    u = action[:, 0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[0, dt], y0=state[:, 0], method='RK45')
    
    return np.vstack(soln.y[:, -1])

# Compute continuous A and B for LQR policy
x_bar = x_e
y_bar = y_e
z_bar = z_e

continuous_A = np.array([
    [-sigma, sigma, 0],
    [rho - z_bar, -1, 0],
    [y_bar, x_bar, -beta]
])
continuous_B = np.array([
    [1],
    [0],
    [0]
])

W, V = np.linalg.eig(continuous_A)

print(f"Eigenvalues of continuous A:\n{W}\n")
print(f"Eigenvectors of continuous A:\n{V}\n")