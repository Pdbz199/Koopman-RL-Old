"""
Make sure you are in this file's directory, then you can run:
python test_env.py --env_name=DoubleWell-v0
"""

import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')
from linear_system.dynamics_env import LinearSystem
from lorenz.dynamics_env import Lorenz
from fluid_flow.dynamics_env import FluidFlow
from double_well.dynamics_env import DoubleWell
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env_name', default="LinearSystem-v0",
                    help='Gym environment (default: LinearSystem-v0)')
args = parser.parse_args()

if args.env_name == "DoubleWell-v0":
    is_3d_env = False
else: is_3d_env = True

# Create the environment
env = gym.make(args.env_name)

# Set up the figure and axis
fig = plt.figure()
if is_3d_env:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)

# Initialize the plot elements
line, = ax.plot([], [], lw=2)
ax.set_xlim(env.state_minimums[0, 0], env.state_maximums[0, 0])
ax.set_ylim(env.state_minimums[1, 0], env.state_maximums[1, 0])
if is_3d_env:
    ax.set_zlim(env.state_minimums[2, ], env.state_maximums[2, 0])
ax.set_xlabel("X")
ax.set_ylabel("Y")
if is_3d_env:
    ax.set_zlabel("Z")
ax.set_title(f"{args.env_name} System Trajectory")

def init():
    line.set_data([], [])
    if is_3d_env:
        line.set_3d_properties([])
    return line,

state, _ = env.reset()
states = [state]
def animate(i):
    next_state, _, _, _, _ = env.step(np.array([0]))
    states.append(next_state)

    line.set_data(np.array(states)[:, 0], np.array(states)[:, 1])
    if is_3d_env:
        line.set_3d_properties(np.array(states)[:, 2])

ani = FuncAnimation(fig, animate, init_func=init, frames=2000, interval=1, repeat=False)
plt.show()