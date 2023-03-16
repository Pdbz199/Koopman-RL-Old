import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.integrate import odeint#, solve_ivp
import sys

sys.path.append('../../../')

from final.control.fluid_flow.dynamics import (
    continuous_A,
    continuous_B,
    continuous_f,
    dt,
    f,
    get_random_initial_conditions,
    state_dim,
    state_maximums,
    state_minimums,
    zero_policy
)

def linear_continuous_f(t, x, u):
    return continuous_A @ x + continuous_B @ u

if __name__ == "__main__":
    num_paths = 100
    num_time_units = 10
    num_steps_per_path = int(num_time_units / dt)

    nonlinearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))
    linearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')

    # On limit cycle
    initial_states = get_random_initial_conditions(num_samples=num_paths)
    # Fixed point
    # initial_states = np.ones((num_paths, state_dim)) * 0.05
    # Random points
    # initial_states = np.random.uniform(
    #     state_minimums,
    #     state_maximums,
    #     [state_dim, num_paths]
    # ).T

    #%% Load LQR policy
    with open('./analysis/tmp/lqr/policy.pickle', 'rb') as handle:
        lqr_policy = pickle.load(handle)

    # Arrays for storing state derivatives
    state_dots = np.zeros((num_paths, num_steps_per_path, state_dim))
    state_dot_hats = np.zeros((num_paths, num_steps_per_path, state_dim))

    # Arrays for storing states
    nonlinearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))
    linearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))

    for path_num in range(num_paths):
        # Get initial states
        nonlinearized_state = np.vstack(initial_states[path_num])
        linearized_state = initial_states[path_num]

        for step_num in range(num_steps_per_path):
            # Store state data
            nonlinearized_states[path_num, step_num] = nonlinearized_state[:, 0]
            linearized_states[path_num, step_num] = linearized_state

            # Get new action
            # nonlinearized_action = zero_policy(nonlinearized_state)
            # linearized_action = zero_policy(linearized_state)
            nonlinearized_action = lqr_policy.get_action(nonlinearized_state)
            linearized_action = lqr_policy.get_action(linearized_state)

            # x_dot from true dynamics
            state_dot = continuous_f(nonlinearized_action[0, 0])(0, nonlinearized_state[:, 0])
            state_dots[path_num, step_num] = state_dot

            # x_dot_hat from linearized dynamics
            state_dot_hat = linear_continuous_f(0, linearized_state, linearized_action)
            state_dot_hats[path_num, step_num] = state_dot_hat[:, 0]

            # Apply action to both systems
            nonlinearized_state = f(nonlinearized_state, nonlinearized_action)
            linearized_state = odeint(
                linear_continuous_f,
                linearized_state,
                t=[0, dt],
                args=(linearized_action[:, 0],),
                tfirst=True
            )[-1]

    # Compute norms between derivative estimates
    state_dot_difference_norms = np.linalg.norm(state_dots - state_dot_hats, axis=2) # (path by step)
    average_state_dot_difference_norm = state_dot_difference_norms.mean()
    print()
    print(
        "Average (state_dot - state_dot_hat) norm:",
        average_state_dot_difference_norm
    )
    average_state_dot_norm = np.linalg.norm(state_dots, axis=2).mean() # (path by step)
    print(
        "Average (state_dot - state_dot_hat) norm normalized by average state_dot norm:",
        average_state_dot_difference_norm / average_state_dot_norm
    )

    # Compute norms between true states and estimated states
    state_difference_norms = np.linalg.norm(nonlinearized_states - linearized_states, axis=2)
    average_state_difference_norm = state_dot_difference_norms.mean()
    print()
    print(
        "Average (nonlinearized state - linearized state) norm:",
        average_state_difference_norm
    )
    average_nonlinearized_state = np.linalg.norm(nonlinearized_states, axis=2).mean()
    print(
        "Average (nonlinearized state - linearized state) norm normalized by average nonlinearized state norm:",
        average_state_difference_norm / average_nonlinearized_state
    )

    # Plot an example trajectory
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)

    episode_num_to_plot = 0
    ax.plot3D(
        nonlinearized_states[episode_num_to_plot, :, 0],
        nonlinearized_states[episode_num_to_plot, :, 1],
        nonlinearized_states[episode_num_to_plot, :, 2]
    )
    ax.plot3D(
        linearized_states[episode_num_to_plot, :, 0],
        linearized_states[episode_num_to_plot, :, 1],
        linearized_states[episode_num_to_plot, :, 2]
    )
    plt.show()