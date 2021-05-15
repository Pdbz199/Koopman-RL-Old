#%%
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
    """ Choosing an action on epsilon-greedy policy """
    return np.argmax(Q_table[state])

def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    """ Temperal difference for updating Q-value of state-action pair """
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

def learning_rate(n: int, min_rate=0.01) -> float:
    """ Decaying learning rate """
    return max(min_rate, min(1.0, 1.0 - math.log10((n+1)/25)))

def exploration_rate(n: int, min_rate=0.1) -> float:
    """ Decaying exploration rate """
    return max(min_rate, min(1, 1.0 - math.log10((n+1)/25)))

# CartPole has been solved! It took 230 episodes
episodes = 100
# rendered_frames = 80
episode_rewards = []
Q_table = np.load('Q_table.npy')
inputs = []
outputs = []

states = []
actions = []
rewards = []

for episode in range(episodes):
    episode_reward = 0
    current_state = env.reset()
    descretized_current_state = discretizer(*current_state)
    done = False

    while done == False:
        states.append(current_state)

        action = policy(descretized_current_state)
        actions.append(action)

        # inputs = np.append(inputs, [[*list(current_state), action]], axis=0) if len(inputs) > 0 else np.array([[*list(current_state), action]])

        # if exploration rate large, tend to try random action
        # if np.random.random() < exploration_rate(episode):
        #     action = env.action_space.sample()

        current_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        new_state = discretizer(*current_state)
        episode_reward += reward

        # outputs = np.append(outputs, [[*list(new_state), policy(new_state)]], axis=0) if len(outputs) > 0 else np.array([[*list(new_state), policy(new_state)]])
        
        # lr = learning_rate(episode)
        # learned_value = new_Q_value(reward, new_state)
        # old_value = Q_table[current_state][action]
        # Q_table[current_state][action] = ((1 - lr) * old_value) + (lr * learned_value)

        descretized_current_state = new_state

        # env.render()
    
    # episode_rewards.append(episode_reward)
    # if len(episode_rewards) == 100:
    #     if np.average(episode_rewards) >= 195.0:
    #         print(f"CartPole has been solved! It took {episode+1} episodes")
    #         break
            
    #     episode_rewards.pop(0)

    # states.pop()

states = np.array(states)
print(states.shape)
actions = np.array(actions)
print(actions.shape)
rewards = np.array(rewards)
print(rewards.shape)
# np.save('optimal-agent/cartpole-states', states)
# np.save('optimal-agent/cartpole-actions', actions)
# np.save('optimal-agent/cartpole-rewards', rewards)
# print(inputs.shape)
# print(outputs.shape)
# np.save('state-action-inputs', inputs)
# np.save('state-action-outputs', outputs)
# np.save('Q_table', Q_table)
env.close()
# %%
