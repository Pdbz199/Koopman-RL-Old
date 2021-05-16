#%%
import gym
import numpy as np

#%%
env = gym.make('CartPole-v0')

#%%
episodes = 200
steps_per_episode = 100

states = []
states_0 = []
states_1 = []
next_states_0 = []
next_states_1 = []
actions = []
rewards = []

#%%
for episode in range(episodes):
    observation = env.reset()
    for t in range(steps_per_episode):
        # env.render()

        action = env.action_space.sample()

        if action == 0:
            states_0.append(observation)
        else:
            states_1.append(observation)
        states.append(observation)
        actions.append(action)

        observation, reward, done, _ = env.step(action) # take a random action

        if action == 0:
            next_states_0.append(observation)
        else:
            next_states_1.append(observation)

        rewards.append(reward)
env.close()

states = np.array(states)
states_0 = np.array(states_0)
states_1 = np.array(states_1)
actions = np.array(actions)
rewards = np.array(rewards)

np.save('random-agent/cartpole-states', states)
np.save('random-agent/cartpole-states-0', states_0)
np.save('random-agent/cartpole-states-1', states_1)
np.save('random-agent/cartpole-next-states-0', next_states_0)
np.save('random-agent/cartpole-next-states-1', next_states_1)
np.save('random-agent/cartpole-actions', actions)
np.save('random-agent/cartpole-rewards', rewards)

#%%
print(rewards.shape)
print(np.mean(rewards))
# %%
