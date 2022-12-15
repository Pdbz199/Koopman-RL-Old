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

# action_range = 75.0
action_range = 25.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

# Policy that only returns 0
def zero_policy(x=None):
    return np.zeros(action_column_shape)

def random_policy(x=None):
    return np.random.choice(all_actions[0], size=action_column_shape)

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
        ])
        sigma_x = np.array([
            [0.7, x],
            [0, 0.5]
        ])

        column_output = b_x + u #+ sigma_x @ np.random.randn(2,1)
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_timesteps = 100.0
    num_steps = int(num_timesteps / dt)

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [state_dim, 1]
    ).T
    initial_state = initial_states[0]

    states = np.empty((state_dim, num_steps))
    actions = np.empty((action_dim, num_steps))
    state = np.vstack(initial_state)
    for step_num in range(num_steps):
        states[:,step_num] = state[:,0]
        action = zero_policy(state)
        actions[:,step_num] = action[:,0]
        state = f(state, action)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(states[0], states[1])
    ax.plot(states[0,0], states[1,0], marker="o", color='g', markersize=4)
    ax.plot(states[0,-1], states[1,-1], marker="o", color='r', markersize=4)
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax = fig.add_subplot(312)
    ax.plot(np.arange(num_steps), states[0])
    ax.plot(0, states[0,0], marker="o", color='g', markersize=4)
    ax.plot(states.shape[1]-1, states[0,-1], marker="o", color='r', markersize=4)
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax = fig.add_subplot(313)
    ax.plot(np.arange(num_steps), states[1])
    ax.plot(0, states[1,0], marker="o", color='g', markersize=4)
    ax.plot(states.shape[1]-1, states[1,-1], marker="o", color='r', markersize=4)
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    plt.show()