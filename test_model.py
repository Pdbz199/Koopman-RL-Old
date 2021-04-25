#%%
import gym
import observables
import numpy as np
from general_model import GeneratorModel
from cartpole_reward import cartpoleReward

#%%
X = (np.load('random-cartpole-states.npy'))[:2].T # states
U = (np.load('random-cartpole-actions.npy'))[:2].T # actions
psi = observables.monomials(2)

#%%
env = gym.make('CartPole-v0')

#%%
model = GeneratorModel(psi, cartpoleReward, [0,1])
print(X.shape)
print(U.shape)
model.fit(X, U)

#%%
# print(model.sample_action(0))

#%%
episodes = 100
episode_rewards = []
for episode in range(episodes):
    episode_reward = 0
    state_num = 0
    current_state = env.reset()
    done = False

    while done == False:
        # env.render()

        action = model.sample_action(current_state)

        model.update_model(current_state, action)

        current_state, reward, done, _ = env.step(int(np.around(action)))
        episode_reward += reward
    model.update_policy(5)
    
    print(f"episode {episode+1} reward:", episode_reward)
    episode_rewards.append(episode_reward)

print("\naverage reward:", np.mean(episode_rewards))

# %%
