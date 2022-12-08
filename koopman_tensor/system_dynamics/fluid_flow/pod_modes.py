# Imports
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint, solve_ivp

# Variables
state_dim = 3
action_dim = 1

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

# Default basic policies
def zero_policy(x=None):
    return np.zeros(action_column_shape)

# Dynamics
omega = 1.0
mu = 0.1
A = -0.1
lamb = 1

dt = 0.01

# Define the fluid flow system
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

# Set the initial values and time step
dt = 0.01
num_timesteps = 100.0
step_cnt = int(num_timesteps / dt)

# Set initial state values
initial_state = np.array([
    [0.2],
    [0.2],
    [1.0]
])

# Generate path
states = odeint(
    func=continuous_f(None),
    y0=initial_state[:,0],
    t=np.arange(0, num_timesteps, dt),
    tfirst=True
).T

# Plot the path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.sca(ax)
plt.plot(
    states[0],
    states[1],
    states[2]
)
plt.title("Path in System")
plt.show()

# Compute the singular value decomposition of the system's trajectory
U, S, V = np.linalg.svd(states)

# The first three POD modes are given by the first three columns of V
mode1 = V[:, 0]
mode2 = V[:, 1]
mode3 = V[:, 2]

# Plot the POD modes using matplotlib
fig = plt.figure()
ax = fig.add_subplot(121)
plt.sca(ax)
plt.plot(mode1)
plt.plot(mode2)
plt.plot(mode3)

ax = fig.add_subplot(122, projection='3d')
plt.sca(ax)
plt.plot(mode1, mode2, mode3, lw=0.5)
plt.xlabel("Mode 1")
plt.ylabel("Mode 2")
ax.set_zlabel("Mode 3")
plt.title("POD Modes of the Fluid Flow")

plt.tight_layout()
plt.show()