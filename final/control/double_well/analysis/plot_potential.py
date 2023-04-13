""" IMPORTS """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from dynamics import (
    get_random_initial_conditions,
    f,
    state_dim,
    state_minimums,
    state_maximums,
    zero_policy
)

def potential(x):
    """
        Compute the potential energy of a given state.
    
        INPUTS:
            x - Column vector of state.

        OUTPUTS:
            Float representing the potential energy.
    """

    return (x[0]**2 - 1)**2 + x[1]**2

""" PLOT DYNAMICS """

if __name__ == "__main__":
    initial_conditions = get_random_initial_conditions(num_samples=1)
    x = np.vstack(initial_conditions[0])

    num_steps = 10000

    xs = np.zeros((num_steps, state_dim))
    potentials = np.zeros(num_steps)

    for step_num in range(num_steps):
        xs[step_num] = x[:, 0]
        potentials[step_num] = potential(x)

        u = zero_policy(x)
        x = f(x, u)

    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
        ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
        ax.set_zlim(0, 2)
        ax.plot(xs[:frame, 0], xs[:frame, 1], potentials[:frame])
        
        if frame == len(xs)-1: print("DONE")

    ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=50, repeat=False)
    plt.show()