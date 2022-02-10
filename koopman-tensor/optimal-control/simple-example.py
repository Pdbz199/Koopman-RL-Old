#%% Imports
import numpy as np
np.random.seed(13)

from enum import IntEnum

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import algorithmsv2_parallel as algorithmsv2
import estimate_L
import utilities

#%% Constants
TWO_THIRDS = 2/3

#%% IntEnum for high/low, up/down
class State(IntEnum):
    HIGH = 1
    LOW = 0

class Action(IntEnum):
    UP = 1
    DOWN = 0

#%% System dynamics
def f(x, u):
    random = np.random.rand()
    if x == State.HIGH and u == Action.UP:
        return State.HIGH if random <= TWO_THIRDS else State.LOW
    elif x == State.HIGH and u == Action.DOWN:
        return State.LOW if random <= TWO_THIRDS else State.HIGH
    elif x == State.LOW and u == Action.UP:
        return State.HIGH if random <= TWO_THIRDS else State.LOW
    elif x == State.LOW and u == Action.DOWN:
        return State.LOW if random <= TWO_THIRDS else State.HIGH

#%% Define cost
def cost(x, u):
    if x == State.HIGH and u == Action.UP:
        return 1.0
    elif x == State.HIGH and u == Action.DOWN:
        return 100.0
    elif x == State.LOW and u == Action.UP:
        return 100.0
    elif x == State.LOW and u == Action.DOWN:
        return 1.0

def costs(xs, us):
    costs = np.empty((xs.shape[1],us.shape[1]))
    for i in range(xs.shape[1]):
        x = np.vstack(xs[:,i])
        for j in range(us.shape[1]):
            u = np.vstack(us[:,j])
            costs[i,j] = cost(x, u)

    return costs

#%% Construct snapshots of u from random agent and initial states x0
N = 100
X = np.random.choice(list(State), size=(1,N))
U = np.random.choice(list(Action), size=(1,N))

#%% Construct snapshots of states following dynamics f(x,u) -> x'
Y = np.empty_like(X)
for i in range(X.shape[1]):
    x = np.vstack(X[:,i])
    u = np.vstack(U[:,i])
    Y[:,i] = f(x, u)

#%% Dictionaries
distinct_xs = 2
distinct_us = 2

enumerated_states = np.array([State.HIGH, State.LOW])
enumerated_actions = np.array([Action.UP, Action.DOWN])

def phi(x):
    phi_x = np.zeros((distinct_us,x.shape[1]))
    phi_x[x[0].astype(int),np.arange(0,x.shape[1])] = 1
    return phi_x

def psi(u):
    psi_u = np.zeros((distinct_us,u.shape[1]))
    psi_u[u[0].astype(int),np.arange(0,u.shape[1])] = 1
    return psi_u

#%% Koopman tensor
tensor = KoopmanTensor(X, Y, U, phi, psi)

#%% Training error
norms = np.empty((N))
for i in range(N):
    phi_x = np.vstack(tensor.Phi_X[:,i]) # current (lifted) state

    action = np.vstack(U[:,i])

    true_x_prime = np.vstack(Y[:,i])
    predicted_x_prime = tensor.B.T @ tensor.K_(action) @ phi_x

    # Compute norms
    norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
print("Average training error:", np.mean(norms))

#%% Testing error
num_episodes = 100
num_steps_per_episode = 100

norms = np.zeros((num_episodes))
for episode in range(num_episodes):
    x = np.array([[np.random.choice(list(State))]])

    for step in range(num_steps_per_episode):
        phi_x = phi(x) # apply phi to state

        action = np.array([[np.random.choice(list(Action))]]) # sample random action

        true_x_prime = np.array([[f(x, action)]])
        predicted_x_prime = tensor.B.T @ tensor.K_(action) @ phi_x

        norms[episode] += utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime
print("Average testing error per episode:", np.mean(norms))

#%% Discretize all controls
All_U = np.array([[Action.UP, Action.DOWN]])

#%% Learn control
algos = algorithmsv2.algos(
    X,
    All_U,
    np.array([0,1]),
    phi,
    psi,
    tensor.K,
    costs,
    beta=0.5,
    epsilon=1e-2,
    bellmanErrorType=0,
    weightRegularizationBool=0,
    u_batch_size=30,
    learning_rate=1e-1
)
# algos.w = np.load('bellman-weights.npy')
algos.w = np.array([[9],[300]])
# algos.w = np.array([[14241],[14241]])
print("Weights before updating:", algos.w)
bellmanErrors, gradientNorms = algos.algorithm2(batch_size=64)
print("Weights after updating:", algos.w)

#%% Construct policy
All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis = algos.pis(x)[:,0]
    # Select action column at index sampled from policy distribution
    u = np.vstack(
        All_U[:,np.random.choice(All_U_range, p=pis)]
    )
    return u

#%% Test policy by simulating system
num_episodes = 100
num_steps_per_episode = 100

costs = np.empty((num_episodes))
for episode in range(num_episodes):
    x = np.array([[np.random.choice(list(State))]])

    cost_sum = 0
    for step in range(num_steps_per_episode):
        u = policy(x)

        cost_sum += cost(x, u)

        x = np.array([[f(x, u)]])

        # if step%250 == 0:
        #     print("Current x:", x)
    costs[episode] = cost_sum
print("Mean cost per episode:", np.mean(costs)) # Cost should be minimized

#%%