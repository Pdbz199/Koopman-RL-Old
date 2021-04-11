#%%
import gym
import numpy as np

#%%
env = gym.make('CartPole-v0')

#%%
episodes = 200
steps_per_episode = 100

states = []
actions = []
rewards = []

#%%
current_state = env.reset()
for episode in range(episodes):
    observation = env.reset()
    for t in range(steps_per_episode):
        # env.render()

        action = env.action_space.sample()

        states.append(current_state)
        actions.append(action)

        new_state, reward, done, _ = env.step(action) # take a random action
        rewards.append(reward)

        current_state = new_state
env.close()

states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)

np.save('random-cartpole-states', states)
np.save('random-cartpole-actions', actions)
np.save('random-cartpole-rewards', rewards)

#%%
print(rewards.shape)
print(np.sum(rewards))
# %%
