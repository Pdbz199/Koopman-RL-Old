#%%
import gym
import observables
import numpy as np
from general_model import GeneratorModel
from cartpole_reward import cartpoleReward

#%%
X = (np.load('random-cartpole-states.npy'))[:5000].T # states
U = (np.load('random-cartpole-actions.npy'))[:5000].T # actions
psi = observables.monomials(2)

#%%
env = gym.make('CartPole-v0')

#%%
model = GeneratorModel(psi, cartpoleReward)
print(X.shape)
print(U.shape)
print(X[:,:2].shape)
print(U[:2].shape)
model.fit(X[:,:2], U[:2])

#%%
print(model.sample_action(0))

#%%
episodes = 150
episode_rewards = []
for episode in range(episodes):
    episode_reward = 0
    state_num = 0
    current_state = env.reset()
    done = False

    while done == False:
        env.render()

        action = model.sample_action(current_state)

        model.update(current_state, action)

        current_state, reward, done, _ = env.step(int(np.around(action)))
        episode_reward += reward
    
    print("episode reward:", episode_reward)
    episode_rewards.append(episode_reward)

print("\naverage reward:", np.mean(episode_rewards))

# %%
