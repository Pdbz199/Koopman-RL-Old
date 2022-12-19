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
        # sigma_x = np.array([
        #     [0.7, x],
        #     [0, 0.5]
        # ])

        column_output = b_x + u #+ sigma_x @ np.random.normal(loc=0, scale=1, size=(2,1))
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
    sigma_x = np.array([
        [0.7, state[0,0]],
        [0, 0.5]
    ])

    drift = np.vstack(continuous_f(u)(0, state[:,0])) * dt
    diffusion = sigma_x @ np.random.normal(loc=0, scale=1, size=(2,1)) * np.sqrt(dt)
    return drift + diffusion

    # soln = solve_ivp(fun=continuous_f(u), t_span=[0, dt], y0=state[:,0], method='RK45')

    # return np.vstack(soln.y[:,-1])

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
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import sys
    try:
        seed = int(sys.argv[1])
    except:
        seed = 123
    np.random.seed(seed)
    sys.path.append('../../../final')
    import observables
    from tensor import KoopmanTensor

    num_timesteps = 30.0
    # num_timesteps = 1.0
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

    X = states[:,:-1]
    Y = np.roll(states, -1, axis=1)[:,:-1]
    U = actions
    koopman_tensor = KoopmanTensor(
        X,
        Y,
        U,
        phi=observables.monomials(state_order),
        psi=observables.monomials(action_order),
        regressor='ols',
        is_generator=True,
        dt=dt
    )
    K_0 = koopman_tensor.K_(np.array([[0]]))
    P_0 = K_0.T #! May not be correct way of getting Perron–Frobenius operator
    k_eigen_values, k_eigen_vectors = np.linalg.eig(K_0)
    p_eigen_values, p_eigen_vectors = np.linalg.eig(P_0)

    xs = np.linspace(-2,2,100)
    ys = np.linspace(-1,1,100)
    XX, YY = np.meshgrid(xs, ys)

    potential = (XX**2 - 1)**2 + YY**2
    
    plt.title("Potential Given [x_0,x_1]")
    plt.contourf(XX, YY, potential)
    plt.xlim(-2,2)
    plt.ylim(-1,1)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Potential Given [x_0,x_1]")
    ax.contourf3D(XX, YY, potential, 50)
    # ax.plot_surface(XX, YY, potential, cmap='viridis')
    ax.set_xlim(-2,2)
    ax.set_ylim(-1,1)
    ax.set_zlim(0,1)
    ax = fig.add_subplot(122, projection='3d')
    XXX, YYY = np.meshgrid(states[0,:100], states[1,:100])
    potential = (XXX**2 - 1)**2 + YYY**2
    ax.contourf3D(XXX, YYY, potential, 50, cmap='binary')
    ax.set_xlim(-2,2)
    ax.set_ylim(-1,1)
    ax.set_zlim(0,1)
    plt.show()

    k_eigen_function_0 = np.empty((xs.shape[0], ys.shape[0]))
    k_eigen_function_1 = np.empty_like(k_eigen_function_0)
    p_eigen_function_0 = np.empty_like(k_eigen_function_0)
    p_eigen_function_1 = np.empty_like(k_eigen_function_0)
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            state = np.array([[xs[i]],[ys[j]]])
            phi_state = koopman_tensor.phi(state)
            k_eigen_function_0[i,j] = (k_eigen_vectors[:,0] @ phi_state)[0]
            k_eigen_function_1[i,j] = (k_eigen_vectors[:,1] @ phi_state)[0]
            p_eigen_function_0[i,j] = (p_eigen_vectors[:,0] @ phi_state)[0]
            p_eigen_function_1[i,j] = (p_eigen_vectors[:,1] @ phi_state)[0]

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].set_title("Koopman Eigenfunction 0")
    axs[0,0].contourf(XX, YY, k_eigen_function_0)
    axs[0,1].set_title("Koopman Eigenfunction 1")
    axs[0,1].contourf(XX, YY, k_eigen_function_1)
    axs[1,0].set_title("Perron Eigenfunction 0")
    axs[1,0].contourf(XX, YY, p_eigen_function_0)
    axs[1,1].set_title("Perron Eigenfunction 1")
    axs[1,1].contourf(XX, YY, p_eigen_function_1)
    plt.tight_layout()
    plt.show()

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
    ax = fig.add_subplot(313)
    ax.plot(np.arange(num_steps), states[1])
    ax.plot(0, states[1,0], marker="o", color='g', markersize=4)
    ax.plot(states.shape[1]-1, states[1,-1], marker="o", color='r', markersize=4)
    plt.show()

    # PLOT STATES OVER TIME
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [])
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax.set_zlim(0, 1)
    # plt.show()

    # Choose the FPS and the number of seconds to run for
    # fps = 60
    # num_seconds = 30

    # First set up the figure, the axis, and the plot element we want to animate
    # fig = plt.figure(figsize=(8,4))
    # plt.axis("off")

    # a = lqr_snapshots[0]
    # a = koopman_snapshots[0]
    # im = plt.imshow(a, cmap='hot', clim=(-1,1))

    # def animate(i):
    #     xs = states[0,:i]
    #     ys = states[1,:i]
    #     zs = (xs**2 - 1)**2 + ys**2
    #     line.set_xdata(xs)
    #     line.set_ydata(ys)
    #     # line.set_3d_properties(np.arange(states.shape[1])[:i])
    #     line.set_3d_properties((xs**2 - 1)**2 + ys**2)
    #     return line,

    # anim = FuncAnimation(
    #     fig,
    #     animate,
    #     frames = num_seconds * fps,
    #     interval = 1000 / fps # in ms
    # )
    
    # plt.show()