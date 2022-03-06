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

#%% Define Q and R
Q = np.eye(4) # np.eye(A_eq.shape[0])
R = 1 # np.eye(2)

#%% Construct datasets
np.random.seed(123)

num_episodes = 100
num_steps_per_episode = 200

X = np.zeros([
    env.observation_space.sample().shape[0],
    num_episodes*num_steps_per_episode+1
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
    X[:, episode*num_steps_per_episode] = x
    for step in range(num_steps_per_episode):
        u = env.action_space.sample() # Sampled from random agent
        U[:, (episode*num_steps_per_episode)+step] = u
        y = env.step(u)[0]
        X[:, (episode*num_steps_per_episode)+(step+1)] = y
        Y[:, (episode*num_steps_per_episode)+step] = y

X = np.array(X)[:, :-1]
Y = np.array(Y)
U = np.array(U)

#%% 
tensor = KoopmanTensor(X, Y, U)

#%% Define cost
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    mat = np.vstack(np.diag(x.T @ Q @ x)) + np.power(u, 2)*R
    return mat

#%% Learn control
u_bounds = np.array([[0, 1]])
All_U = np.array([[0, 1]])
gamma = 0.5
lamb = 1.0
lr = 1e-1
epsilon = 1e-3

algos = algorithmsv2.algos(
    X,
    All_U,
    u_bounds[0],
    tensor,
    cost,
    gamma=gamma,
    epsilon=epsilon,
    bellman_error_type=0,
    learning_rate=lr,
    weight_regularization_bool=True,
    weight_regularization_lambda=lamb,
    optimizer='adam'
)

algos.w = np.load('bellman-weights.npy') # Current Bellman error: 841.875344129739
# print("Weights before updating:", algos.w)
# bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
# print("Weights after updating:", algos.w)

#%% Extract policy
All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis = algos.pis(x)[:,0]
    # Select action column at index sampled from policy distribution
    u_ind = np.random.choice(All_U_range, p=pis)
    u = np.vstack(All_U[:,u_ind])
    return u[0,0]

def random_policy(x):
    return env.action_space.sample()

#%% Test policy by simulating system
num_episodes = 10
rewards = np.zeros([num_episodes])
for episode in range(num_episodes):
    x = np.vstack(env.reset())
    done = False
    while not done:
        env.render()
        u = policy(x)

        next_state,reward,done,info = env.step(u)
        rewards[episode] += reward

        x = np.vstack(next_state)
print("Mean reward per episode:", np.mean(rewards))

#%%