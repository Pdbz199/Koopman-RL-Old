from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time, math, random
from typing import Tuple

import gym
env = gym.make('CartPole-v0')
# observation space: Box([position of cart, velocity of cart, angle of pole, rotation rate of pole]])
# print(env.observation_space)
# action space: Discrete(left, right)
# print(env.action_space)

n_bins = ( 6, 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def discretizer( _, __, angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continuous state into a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([ lower_bounds, upper_bounds ])
    return tuple( map( int, est.transform([[ angle, pole_velocity ]])[0] ) )

Q_table = np.zeros(n_bins + (env.action_space.n,))

def policy(state: tuple):
    """Choosing an action on epsilon-greedy policy"""
    return np.argmax(Q_table[state])

def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    """Temperal difference for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n+1)/25)))

def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n+1)/25)))

episodes = 1
# rendered_frames = 80
Q_table = np.load('Q_table.npy')
for episode in range(episodes):
    current_state = discretizer(*env.reset())
    done = False

    while done==False:
        action = policy(current_state)

        # if exploration rate large, tend to try random action
        # if np.random.random() < exploration_rate(episode):
        #     action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        new_state = discretizer(*observation)
        
        # lr = learning_rate(episode)
        # learned_value = new_Q_value(reward, new_state)
        # old_value = Q_table[current_state][action]
        # Q_table[current_state][action] = ((1 - lr) * old_value) + (lr * learned_value)

        current_state = new_state

        if episode == 0:
            env.render()

# np.save('Q_table', Q_table)
env.close()