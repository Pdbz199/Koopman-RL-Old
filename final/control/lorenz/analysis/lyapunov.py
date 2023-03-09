import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.integrate import odeint

sys.path.append('../../../')

from final.control.lorenz.dynamics import (
    continuous_f,
    dt,
    f,
    get_random_initial_conditions,
    state_dim,
    zero_policy
)

if __name__ == "__main__":
    num_time_units = 100
    num_steps_per_path = int(num_time_units / dt)

    states = np.zeros((num_steps_per_path, state_dim))

    initial_states = get_random_initial_conditions(num_samples=1)

    state = np.vstack(initial_states[0])
    for step_num in range(num_steps_per_path):
        states[step_num] = state[:, 0]
        state = f(state, zero_policy(state))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    img = ax.scatter(
        states[:, 0],
        states[:, 1],
        states[:, 2],
        c=np.arange(0, num_time_units, dt),
        cmap=plt.hot(),
        s=0.3
    )
    fig.colorbar(img)
    plt.show()

    # library = [states[1]]
    # for step_num in range(2, num_steps_per_path):
    #     if np.any(np.linalg.norm(states[step_num] - library, axis=1) > 0.1):
    #         library.append(states[step_num])
    # library = np.array(library)

    # for state in library:
    #     closest_state_index = np.argmin(np.linalg.norm(state - library, axis=1))
    #     closest_state = library[closest_state_index]
    #     state_delta = state - closest_state
    #     epsilon = 0.01
    #     state_epsilon = state + epsilon * state_delta

    #     T = 1
    #     state_epsilon_T = odeint(continuous_f(), state_epsilon, t=[0, T / dt], tfirst=True)[-1]
    #     state_T = odeint(continuous_f(), state, t=[0, T / dt], tfirst=True)[-1]

    #     sigma = 1 / T * np.log(np.linalg.norm(state_epsilon_T - state_T) / epsilon)
    #     print(sigma)

    state = initial_states[0]
    perturbed_state = state + np.random.uniform(low=-0.1, high=0.1, size=(3))

    state_T_path = odeint(
        continuous_f(),
        y0=state,
        t=np.arange(0, num_time_units, dt),
        tfirst=True
    ).T
    perturbed_state_T_path = odeint(
        continuous_f(),
        y0=perturbed_state,
        t=np.arange(0, num_time_units, dt),
        tfirst=True
    ).T

    separation = np.linalg.norm(perturbed_state_T_path - state_T_path, axis=0)

    lyapunov_exponent = np.mean(np.log(separation[1:] / separation[:-1])) / dt

    print(f"Starting from {state} (and {perturbed_state}):")
    print(f"  The Lyapunov exponent is {lyapunov_exponent}")