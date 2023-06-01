import gym
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces

from lorenz.cost import cost
from lorenz.dynamics import (
    action_dim,
    action_minimums,
    action_maximums,
    dt,
    f,
    state_dim,
    state_maximums,
    state_minimums
)
from gym.envs.registration import register

max_episode_steps = int(20 / dt)

register(
    id='Lorenz-v0',
    entry_point='lorenz.dynamics_env:Lorenz',
    max_episode_steps=max_episode_steps
)

class Lorenz(gym.Env):
    def __init__(self):
        # Observations are 3-dimensional vectors indicating spatial location.
        self.observation_space = spaces.Box(
            low=state_minimums[:, 0],
            high=state_maximums[:, 0],
            shape=(state_dim,),
            dtype=np.float64
        )

        # We have a continuous action space. In this case, there is only 1 dimension per action
        self.action_space = spaces.Box(
            low=action_minimums[:, 0],
            high=action_maximums[:, 0],
            shape=(action_dim,),
            dtype=np.float64
        )

        self.states = []

        self.render_has_been_called = False

    def reset(self, seed=None, options={"state": None}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the initial state uniformly at random
        self.state = self.observation_space.sample() if options['state'] is None else options['state']
        self.states = [self.state]

        # Track number of steps taken
        self.step_count = 0

        # Rendering not set up by default
        self.render_has_been_called = False

        return self.state, {}

    @staticmethod
    def reward(state, action):
        return -cost(
            np.vstack(state),
            np.vstack(action)
        )[0, 0]

    def step(self, action):
        # Compute reward of system
        reward = Lorenz.reward(self.state, action)

        # Update state
        self.state = f(
            np.vstack(self.state),
            np.vstack(action)
        )[:, 0]
        self.states.append(self.state)

        # Update global step count
        self.step_count += 1

        # An episode is done if the system has run for max_episode_steps
        terminated = self.step_count >= max_episode_steps

        return self.state, reward, terminated, False, {}

    def render(self):
        if not self.render_has_been_called:
            plt.close('all')
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
            plt.show(block=False)

            self.render_has_been_called = True

        states = np.array(self.states)
        plt.cla()
        self.ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
        self.ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
        self.ax.set_zlim(state_minimums[2, 0], state_maximums[2, 0])
        self.ax.plot(states[:, 0], states[:, 1], states[:, 2])
        self.fig.canvas.draw()
        plt.pause(0.001)