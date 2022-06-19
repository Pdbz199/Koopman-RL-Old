#%% Imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

#%% Define initial parameters
init_c = 0

#%% System dynamics
state_dim = 1
action_dim = 1

state_column_shape = [state_dim,1]
action_column_shape = [action_dim,1]

sigma = 10
rho = 28
beta = 8/3

action_range = np.array([-50, 50])
step_size = 0.1
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

def continuous_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - state vector
            t - timestep
        """
        x, y, z = input

        x_dot = sigma * ( y - x )
        y_dot = ( rho - z ) * x - y
        z_dot = x * y - beta * z

        u = action
        if u is None:
            u = np.random.choice(all_actions, size=action_column_shape)

        return [ x_dot + u, y_dot, z_dot ]

    return f_u

t_span = np.arange(0, 0.01, 0.001)
t_span_range = np.array([0, 0.5])
def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

# Create the figure
# fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
initial_state = np.array([[0], [1], [1.05]])
y = initial_state
state = initial_state
pushed_forward_states = np.zeros([10000,3])
for i in range(10000):
    state = f(state, np.array([[0]]))
    pushed_forward_states[i] = state[:,0]

line, = ax.plot3D(
    pushed_forward_states[:,0],
    pushed_forward_states[:,1],
    pushed_forward_states[:,2],
    lw=0.75
)
point, = ax.plot3D(
    initial_state[0],
    initial_state[1],
    initial_state[2],
    'r.',
    markersize=10
)

ax.set_xlabel('x')
ax.set_xlim(-20.0, 20.0)
ax.set_ylim(-50.0, 50.0)
ax.set_zlim(0.0, 50.0)

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a vertically oriented slider to control the beta.
ax_c = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor=axcolor)
c_slider = Slider(
    ax=ax_c,
    label="action",
    valmin=-50,
    valmax=50,
    valinit=init_c,
    orientation="vertical"
)

#%% The function to be called anytime a slider's value changes
def update(val):
    global y
    u = c_slider.val
    y = f(np.vstack(y), np.array([[u]]))
    
    point.set_xdata(y[0])
    point.set_ydata(y[1])
    point.set_3d_properties(y[2])
    
    fig.canvas.draw_idle()

c_slider.on_changed(update)

#%% Handle ctrl+c
def signal_handler(sig, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

#%% Run loop
while(True):
    update(0)
    plt.pause(0.01)