import matplotlib.pyplot as plt
import numpy as np
import sys

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from scipy.integrate import odeint

sys.path.append('./')
from dynamics import (
    continuous_A,
    continuous_B,
    dt,
    state_dim,
    state_maximums,
    state_minimums,
    zero_policy
)

def continuous_uncontrolled_dynamics(t, x):
    _x = np.vstack(x)
    u = zero_policy(_x)
    return (continuous_A @ _x + continuous_B @ u)[:, 0]

num_episodes = 50

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
).T

states = []
for episode_num in range(num_episodes):
    print(f"Episode #{episode_num}", end='\r')
    soln = odeint(
        continuous_uncontrolled_dynamics,
        initial_states[episode_num],
        t=np.arange(0, 20, dt),
        tfirst=True
    )
    states.append(soln)
states = np.array(states)

episode_to_plot = 15
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
plt.sca(ax)
for state_dim_num in range(state_dim):
    plt.plot(states[episode_to_plot, :, state_dim_num])

ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.plot3D(
    states[episode_to_plot, :, 0],
    states[episode_to_plot, :, 1],
    states[episode_to_plot, :, 2]
)
ax.scatter3D(
    states[episode_to_plot, 0, 0],
    states[episode_to_plot, 0, 1],
    states[episode_to_plot, 0, 2],
    color='green'
)
ax.scatter3D(
    states[episode_to_plot, -1, 0],
    states[episode_to_plot, -1, 1],
    states[episode_to_plot, -1, 2],
    color='red'
)

plt.show()