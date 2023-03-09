import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../../')

from final.control.lorenz.dynamics import (
    dt,
    f,
    state_dim,
    state_minimums,
    state_maximums,
    zero_policy
)

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
    ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
    ax.set_zlim(state_minimums[2, 0], state_maximums[2, 0])

    num_time_units = 10
    num_steps = int(num_time_units / dt)
    states = np.zeros((num_steps, state_dim))
    dynamics_line, = ax.plot([], [], [], 'b-', lw=2)

    state = np.vstack([0.2, 0.2, 0.2])

    def animate(frame_num):
        global state

        states[frame_num] = state[:, 0]
        dynamics_line.set_data(states[:frame_num, 0], states[:frame_num, 1])
        dynamics_line.set_3d_properties(states[:frame_num, 2])

        action = zero_policy(state)
        state = f(state, action)

        return dynamics_line

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=num_steps,
        interval=1
    )
    plt.show()