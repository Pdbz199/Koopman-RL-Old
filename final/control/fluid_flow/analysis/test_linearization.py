import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint#, solve_ivp
import sys

sys.path.append('../../../')

from final.control.fluid_flow.dynamics import (
    continuous_A,
    continuous_B,
    dt,
    f,
    get_random_initial_conditions,
    state_dim
)

def linear_continuous_f(t, x, u):
    return continuous_A @ x + continuous_B @ u

if __name__ == "__main__":
    num_paths = 100
    num_time_units = 10
    num_steps_per_path = int(num_time_units / dt)

    nonlinearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))
    linearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # initial_states = get_random_initial_conditions(num_samples=num_paths)
    initial_states = [[0.4, 0.4, 0.4]]

    for path_num in range(num_paths):
        nonlinearized_state = np.vstack(initial_states[path_num])

        zero_action = np.array([[0]])

        for step_num in range(num_steps_per_path):
            nonlinearized_states[path_num, step_num] = nonlinearized_state[:, 0]
            nonlinearized_state = f(nonlinearized_state, zero_action)

        linearized_states[path_num] = odeint(
            linear_continuous_f,
            initial_states[path_num],
            t=np.arange(0, num_time_units, dt),
            args=(zero_action[:, 0],),
            tfirst=True
        )

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