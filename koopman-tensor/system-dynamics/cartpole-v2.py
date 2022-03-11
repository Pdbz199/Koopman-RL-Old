#%% Imports
import gym
import numpy as np

from sklearn.kernel_approximation import RBFSampler

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% Load environment
env = gym.make('CartPole-v0')

#%% Construct dataset
np.random.seed(123)

num_episodes = 400
num_steps_per_episode = 100

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

#%% Compute koopman tensor
rbf_state_feature = RBFSampler(gamma=1, n_components=500, random_state=1)
rbf_state_feature.fit(X.T)
rbf_action_feature = RBFSampler(gamma=1, n_components=500, random_state=1)
rbf_action_feature.fit(U.T)

def phi(x):
    """ x must be one or a set column vectors """
    entry_0 = np.vstack(x[:,0])
    assert entry_0.shape[0] >= entry_0.shape[1]
    return rbf_state_feature.transform(x.T).T

def psi(u):
    """ u must be one or a set of column vectors """
    entry_0 = np.vstack(u[:,0])
    assert entry_0.shape[0] >= entry_0.shape[1]
    return rbf_action_feature.transform(u.T).T

tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(3),
    psi=observables.monomials(3),
    regressor='sindy'
)

#%% Training error
print("Training error:")
training_norms = np.zeros([num_episodes*num_steps_per_episode])
norms_states = np.empty([num_episodes,num_steps_per_episode])
for i in range(num_episodes*num_steps_per_episode):
    x = np.vstack(X[:, i])
    phi_x = tensor.phi(x)

    true_x_prime = np.vstack(Y[:, i])

    predicted_x_prime = tensor.B.T @ tensor.K_(U[:, i]) @ phi_x

    training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
print("Mean training norm:", np.mean(training_norms))

#%% Testing error
print("Testing error:")
testing_norms = np.zeros([num_episodes,num_steps_per_episode])
norms_states = np.empty([num_episodes,num_steps_per_episode])
for episode in range(num_episodes):
    x = np.vstack(env.reset())
    for step in range(num_steps_per_episode):
        u = env.action_space.sample()

        true_x_prime = np.vstack(env.step(u)[0])
        predicted_x_prime = tensor.B.T @ tensor.K_(np.array([[u]])) @ tensor.phi(x)

        testing_norms[episode, step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
        norms_states[episode,step] = utilities.l2_norm(x, np.zeros_like(x))

        x = true_x_prime
avg_norm_by_path = np.mean(norms_states, axis=1)
print("Mean testing norm:", np.mean(testing_norms))
print("Avg testing error over all episodes normalized by avg norm of state path:", np.mean((np.mean(testing_norms, axis=1)/avg_norm_by_path)))
#%%