import gym
import numpy as np

from gym import spaces

from fluid_flow.cost import cost
from fluid_flow.dynamics import (
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
    id='FluidFlow-v0',
    entry_point='fluid_flow.dynamics_env:FluidFlow',
    max_episode_steps=max_episode_steps
)

class FluidFlow(gym.Env):
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

    def reset(self, seed=None, options={}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the initial state uniformly at random
        self.state = self.observation_space.sample()

        # Track number of steps taken
        self.step_count = 0

        return self.state, {}

    def step(self, action):
        # Compute reward of system
        reward = -cost(
            np.vstack(self.state),
            np.vstack(action)
        )[0, 0]

        # Update state
        self.state = f(
            np.vstack(self.state),
            np.vstack(action)
        )[:, 0]

        # Update global step count
        self.step_count += 1

        # An episode is done if the system has run for max_episode_steps
        terminated = self.step_count >= max_episode_steps

        return self.state, reward, terminated, False, {}