#%% Imports
import numpy as np

from enum import IntEnum

#%% IntEnum for high/low, up/down
class State(IntEnum):
    HIGH = 1
    LOW = -1

class Action(IntEnum):
    UP = 1
    DOWN = -1

#%% System dynamics
def f(x, u):
    random = np.random.rand()
    if x == State.HIGH and u == Action.UP:
        return State.HIGH if random <= 2/3 else State.LOW
    elif x == State.HIGH and u == Action.DOWN:
        return State.LOW if random <= 2/3 else State.HIGH
    elif x == State.LOW and u == Action.UP:
        return State.HIGH if random <= 2/3 else State.LOW
    elif x == State.LOW and u == Action.DOWN:
        return State.LOW if random <= 2/3 else State.HIGH

#%% Define cost
def cost(x, u):
    if x == State.HIGH and u == Action.UP:
        return 0.5
    elif x == State.HIGH and u == Action.DOWN:
        return 0.0
    elif x == State.LOW and u == Action.UP:
        return 0.0
    elif x == State.LOW and u == Action.DOWN:
        return 1

#%%