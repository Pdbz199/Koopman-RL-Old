#%% Imports
import gym
import numpy as np

# from sklearn.kernel_approximation import RBFSampler

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% Load environment
env = gym.make('CartPole-v0')

#%% Construct dataset
seed = 123
env.seed(seed)
np.random.seed(seed)

num_datapoints = 80000
half_datapoints = int(num_datapoints/2)

X = np.zeros([
    env.observation_space.sample().shape[0],
    num_datapoints
])
Y = np.zeros([
    env.observation_space.sample().shape[0],
    num_datapoints
])
U = np.zeros([
    1,
    num_datapoints
])

total_datapoints = 0
while total_datapoints < num_datapoints:
    x = env.reset()
    done = False
    while not done and total_datapoints < num_datapoints:
        X[:, total_datapoints] = x
        u = env.action_space.sample() # Sampled from random agent
        U[:, total_datapoints] = u
        y,reward,done,info = env.step(u)
        Y[:, total_datapoints] = y
        x = y
        total_datapoints += 1

train_X = X[:, :half_datapoints]
train_Y = Y[:, :half_datapoints]
train_U = U[:, :half_datapoints]

test_X = X[:, half_datapoints:]
test_Y = Y[:, half_datapoints:]
test_U = U[:, half_datapoints:]

# num_episodes = 100
# num_steps_per_episode = 200

# X = np.zeros([
#     env.observation_space.sample().shape[0],
#     num_episodes*num_steps_per_episode+1
# ])
# Y = np.zeros([
#     env.observation_space.sample().shape[0],
#     num_episodes*num_steps_per_episode
# ])
# U = np.zeros([
#     1,
#     num_episodes*num_steps_per_episode
# ])

# for episode in range(num_episodes):
#     x = env.reset()
#     X[:, episode*num_steps_per_episode] = x
#     for step in range(num_steps_per_episode):
#         u = env.action_space.sample() # Sampled from random agent
#         U[:, (episode*num_steps_per_episode)+step] = u
#         y = env.step(u)[0]
#         X[:, (episode*num_steps_per_episode)+(step+1)] = y
#         Y[:, (episode*num_steps_per_episode)+step] = y

# X = np.array(X)[:, :-1]

#%% Compute koopman tensor
# rbf_state_feature = RBFSampler(gamma=1, n_components=500, random_state=1)
# rbf_state_feature.fit(X.T)
# rbf_action_feature = RBFSampler(gamma=1, n_components=500, random_state=1)
# rbf_action_feature.fit(U.T)

# def phi(x):
#     """ x must be one or a set column vectors """
#     entry_0 = np.vstack(x[:,0])
#     assert entry_0.shape[0] >= entry_0.shape[1]
#     return rbf_state_feature.transform(x.T).T

# def psi(u):
#     """ u must be one or a set of column vectors """
#     entry_0 = np.vstack(u[:,0])
#     assert entry_0.shape[0] >= entry_0.shape[1]
#     return rbf_action_feature.transform(u.T).T

tensor = KoopmanTensor(
    train_X,
    train_Y,
    train_U,
    phi=observables.monomials(2),
    psi=observables.monomials(2),
    regressor='ols'
)

#%% Training error
print("\nTraining error:")

training_norms = np.zeros([train_X.shape[1]])

for i in range(train_X.shape[1]):
    x = np.vstack(train_X[:, i])
    phi_x = tensor.phi(x)

    predicted_x_prime = tensor.B.T @ tensor.K_(train_U[:, i]) @ phi_x
    true_x_prime = np.vstack(train_Y[:, i])

    training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)

# for i in range(num_episodes*num_steps_per_episode):
#     x = np.vstack(X[:, i])
#     phi_x = tensor.phi(x)

#     true_x_prime = np.vstack(Y[:, i])

#     predicted_x_prime = tensor.B.T @ tensor.K_(U[:, i]) @ phi_x

#     training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)

print("Mean training norm:", np.mean(training_norms))

#%% Testing error
print("\nTesting error:")

testing_norms = np.zeros([test_X.shape[1]])

for i in range(test_X.shape[1]):
    x = np.vstack(test_X[:, i])
    phi_x = tensor.phi(x)

    predicted_x_prime = tensor.B.T @ tensor.K_(test_U[:, i]) @ phi_x
    true_x_prime = np.vstack(test_Y[:, i])

    testing_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)

# Test dynamics by simulating system
# testing_norms = []
# num_episodes = 1300
# for episode in range(num_episodes):
#     x = np.vstack(env.reset())
#     done = False
#     while not done:
#         phi_x = tensor.phi(x)
#         u = env.action_space.sample()

#         predicted_x_prime = tensor.B.T @ tensor.K_(np.array([[u]])) @ phi_x
#         true_x_prime,reward,done,info = env.step(u)
#         true_x_prime = np.vstack(true_x_prime)

#         testing_norms.append(utilities.l2_norm(true_x_prime, predicted_x_prime))

#         x = true_x_prime
# testing_norms = np.array(testing_norms)

# num_episodes = 5000
# num_steps_per_episode = 10

# testing_norms = np.zeros([num_episodes, num_steps_per_episode])
# norms_states = np.zeros([num_episodes, num_steps_per_episode])

# for episode in range(num_episodes):
#     x = np.vstack(env.reset())
#     for step in range(num_steps_per_episode):
#         phi_x = tensor.phi(x)
#         u = env.action_space.sample()

#         predicted_x_prime = tensor.B.T @ tensor.K_(np.array([[u]])) @ phi_x
#         true_x_prime = np.vstack(env.step(u)[0])

#         testing_norms[episode, step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
#         norms_states[episode, step] = utilities.l2_norm(true_x_prime, np.zeros_like(true_x_prime))

#         x = true_x_prime
# avg_norm_by_path = np.mean(norms_states, axis=1)

print("Mean testing norm:", np.mean(testing_norms))
# print("Avg testing error over all episodes normalized by avg norm of state path:", np.mean((np.mean(testing_norms, axis=1)/avg_norm_by_path)))

#%%