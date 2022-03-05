#%% Imports
import gym
import numpy as np

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import algorithmsv2_parallel as algorithmsv2
# import observables

#%% Load environment
env = gym.make('CartPole-v0')

#%% Construct datasets
np.random.seed(123)

num_episodes = 100
num_steps_per_episode = 200

X = np.zeros([
    env.observation_space.sample().shape[0],
    num_episodes*(num_steps_per_episode+1)
])
Y = np.zeros([
    env.observation_space.sample().shape[0],
    num_episodes*num_steps_per_episode
])
U = np.zeros([
    1,
    num_episodes*num_steps_per_episode
])

for episode in range(num_episodes):
    x = env.reset()
    X[:, num_steps_per_episode*episode] = x
    for step in range(num_steps_per_episode):
        u = env.action_space.sample() # Sampled from random agent
        U[:, (num_steps_per_episode*episode)+step] = u
        y = env.step(u)[0]
        X[:, (num_steps_per_episode*episode)+step+1] = y
        Y[:, (num_steps_per_episode*episode)+step] = y

X = np.array(X)[:, :-1]
Y = np.array(Y)
U = np.array(U)

#%% 
tensor = KoopmanTensor(X, Y, U)

#%% Learn control
gamma = 0.5
lamb = 1.0

algos = algorithmsv2.algos(
    X,
    All_U,
    u_bounds[0],
    tensor,
    cost,
    gamma=gamma,
    epsilon=0.001,
    bellman_error_type=0,
    learning_rate=1e-1,
    weight_regularization_bool=True,
    weight_regularization_lambda=lamb,
    optimizer='adam'
)

algos.w = np.load('bellman-weights.npy')
print("Weights before updating:", algos.w)
bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
print("Weights after updating:", algos.w)

#%%