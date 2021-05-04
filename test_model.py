#%%
# import gym
import observables
import numpy as np
from general_model import GeneratorModel
# from cartpole_reward import cartpoleReward
from simple_env import SimpleTestEnv

#%%
# X = (np.load('random-cartpole-states.npy'))[:2].T # states
# U = (np.load('random-cartpole-actions.npy'))[:2].T # actions
X = np.array([[0, 0],
              [0, 0]])
U = np.array([10, 5])
# print(X.shape)
# print(U.shape)
psi = observables.monomials(6)

#%%
env = SimpleTestEnv()

#%%
model = GeneratorModel(psi, env.reward, [-1.0, 1.0])

#%%
model.fit(X, U)

#%%
# print(model.sample_action(0))

#%%
episode_rewards = []
#%%
episodes = 1
for episode in range(episodes):
    episode_reward = 0
    state_num = 0
    current_state = env.reset()
    done = False

    while done == False:
        # env.render()

        action = np.random.uniform(-1.0, 1.0)

        current_state, reward, done, _ = env.step(action)
        episode_reward += reward

        model.update_model(current_state, action)

    model.update_policy(3)
    
    print(f"episode {episode+1} reward:", episode_reward)
    episode_rewards.append(episode_reward)

print("\naverage reward:", np.mean(episode_rewards))

# %%
