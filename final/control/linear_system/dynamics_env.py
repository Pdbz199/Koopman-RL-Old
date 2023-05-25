import gym
import numpy as np
import sys

from gym import spaces

sys.path.append('../../linear_system')
from cost import cost
from dynamics import (
    action_dim,
    action_minimums,
    action_maximums,
    f,
    state_dim,
    state_maximums,
    state_minimums
)
from gym.envs.registration import register

max_episode_steps = 200

register(
    id='LinearSystem-v0',
    entry_point='linear_system.dynamics_env:LinearSystem',
    max_episode_steps=max_episode_steps
)

class LinearSystem(gym.Env):
    def __init__(self):
        # Observations are 3-dimensional vectors indicating spatial location.
        self.observation_space = spaces.Box(low=state_minimums[:, 0], high=state_maximums[:, 0], shape=(state_dim,))

        # We have a continuous action space. In this case, there is only 1 dimension per action
        self.action_space = spaces.Box(low=action_minimums[:, 0], high=action_maximums[:, 0], shape=(action_dim,))

    def _get_obs(self):
        return self.state

    def reset(self, seed=None, options={}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the initial state uniformly at random
        self.state = self.np_random.uniform(low=state_minimums[:, 0], high=state_maximums[:, 0], size=state_dim)

        # Track number of steps taken
        self.step_count = 0

        # Get initial observation and information
        observation = self._get_obs()
        # info = self._get_info()
        info = {}

        return observation, info

    def step(self, action):
        # Compute reward of system
        reward = -cost(np.vstack(self.state), np.vstack(action))[0, 0]

        # Update state
        self.state = f(np.vstack(self.state), np.vstack(action))[:, 0]

        # Get observation and information
        observation = self._get_obs()
        # info = self._get_info()
        info = {}

        # Update global step count
        self.step_count += 1

        # An episode is done if the system has run for max_episode_steps
        terminated = self.step_count >= max_episode_steps

        return observation, reward, terminated, False, info