import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class SimpleTestEnv(gym.Env):
    """
    Description:
        State does not change or progress, the goal is to simply minimize the reward.

    Observation:
        Type: Box(2)
        Num   Observation                 Min                     Max
        0     x_1                         -10                     10
        1     x_2                         -10                     10

    Actions:
        Type: Box(1)
        Num   Action                      Min                     Max
        0     An arbitrary action         -10                     10

    Reward:
        Reward is 1 for action 0, but 1/action for all others

    Starting State:
        State is x = [0, 0]

    Episode Termination:
        If more than 20 steps in a row are taken without selecting action 0
    """

    def __init__(self):
        self.steps_without_correct_action = 0

        self.action_space = spaces.Box(-10, 10, shape=(1,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,)) #, dtype=np.float32

        self.seed()
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def reward(state, action):
        reward = 1.0
        if action != 0.0:
            reward = np.abs(action)/10.0
        return reward

    def step(self, action):
        if action <= -10.0 and action >= 10.0: raise Exception

        reward = self.reward(self.state, action)

        if action != 0.0:
            self.steps_without_correct_action += 1.0

        done = self.steps_without_correct_action >= 20

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            elif self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [0,0]
        self.steps_beyond_done = None
        self.steps_without_correct_action = 0
        return np.array(self.state)