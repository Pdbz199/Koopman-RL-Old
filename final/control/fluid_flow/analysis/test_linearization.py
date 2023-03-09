import matplotlib.pyplot as plt
import numpy as np
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
    num_paths = 10_000
    num_time_units = 10
    num_steps_per_path = int(num_time_units / dt)

    nonlinearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))
    linearized_states = np.zeros((num_paths, num_steps_per_path, state_dim))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # On limit cycle
    # initial_states = get_random_initial_conditions(num_samples=num_paths)
    # Fixed point
    initial_states = np.ones((num_paths, state_dim)) * 0.05
    # Random points
    # initial_states = np.random.uniform(
    #     state_minimums,
    #     state_maximums,
    #     [state_dim, num_paths]
    # ).T

    # Compute state_dots and state_dot_hats
    state_dots = np.zeros((num_paths, state_dim))
    state_dot_hats = np.zeros((num_paths, state_dim))
    for path_num in range(num_paths):
        state = np.vstack(initial_states[path_num])
        zero_action = zero_policy(state)

        # x_dot from true dynamics
        state_dot = continuous_f(zero_action[0, 0])(0, state[:, 0])
        state_dots[path_num] = state_dot

        # x_dot_hat from linearized dynamics
        state_dot_hat = linear_continuous_f(0, state, zero_action)
        state_dot_hats[path_num] = state_dot_hat[:, 0]

    # Compute norms between derivative estimates
    state_dot_difference_norms = np.linalg.norm(state_dots - state_dot_hats, axis=1)
    average_state_dot_difference_norm = state_dot_difference_norms.mean()
    print(
        "Average (state_dot - state_dot_hat) norm:",
        average_state_dot_difference_norm
    )
    average_state_dot = state_dots.mean()
    print(
        "Average (state_dot - state_dot_hat) norm normalized by average state_dot norm:",
        average_state_dot_difference_norm / average_state_dot
    )