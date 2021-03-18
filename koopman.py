import numpy as np
import pykoopman as pk
from pydmd import OptDMD, DMD

'''====================== TESTING ON SIMPLE FUNCTION ======================'''
M = 4
# random initial state?
# potential constant deviation
X = np.zeros((4, M))
X[:,0] = [1,2,3,4] # sample from distribution, add one on it for a while
for val in range(1, M):
    X[:,val] = X[:,val-1] + 1

# Fit Koopman operator using closed-form solution to DMD
optdmd = OptDMD(svd_rank=2)
model_optdmd = pk.Koopman(regressor=optdmd)
model_optdmd.fit(X.T)

test_point = np.array([100,101,102,103])
prediction = model_optdmd.predict(test_point)
print("Prediction:", prediction)
prediction = np.round(np.real(prediction))
expectation = np.array([101,102,103,104], dtype=float)
print("Expectation:", expectation)
print("prediction ~= expectation:", np.array_equal(prediction, expectation))

'''====================== TESTING ON POLICY FUNCTION ======================'''
X = np.load('state-action-inputs.npy')
# X = X[:int(X.shape[0]*0.5)]

# Fit Koopman operator using closed-form solution to DMD
optdmd = OptDMD(svd_rank=15)
model_optdmd = pk.Koopman(regressor=optdmd)
model_optdmd.fit(X)

index = np.random.randint(0, X.shape[0])
print(f"Point {index} of X")
test_point = X[index]
prediction = model_optdmd.predict(test_point)
print("Prediction:", prediction)
prediction = np.round(np.real(prediction))
expectation = X[index+1]
print("Expectation:", expectation)
print("prediction ~= expectation:", np.array_equal(prediction, expectation))

'''====================== TESTING AGAINST GROUND-TRUTH ======================'''
import math
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import gym
env = gym.make('CartPole-v0')

Q_table = np.load('Q_table.npy')

n_bins = ( 6, 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def discretizer( _, __, angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continuous state into a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([ lower_bounds, upper_bounds ])
    return tuple( map( int, est.transform([[ angle, pole_velocity ]])[0] ) )

def policy(state: tuple):
    """ Choosing an action on epsilon-greedy policy """
    return np.argmax(Q_table[state])

num_steps = 100
correctness_arr = np.zeros(num_steps)
# START FROM BEGINNING STATE
current_state = discretizer(*env.reset())
action = policy(current_state)
prediction = model_optdmd.predict(np.array([*list(current_state), action]))
prediction = np.round(np.real(prediction))

for i in range(num_steps):
    # ENV CODE
    observation, reward, done, _ = env.step(action)
    new_state = discretizer(*observation)
    next_action = policy(new_state)

    # PREDICT AND GO ON
    prediction = model_optdmd.predict(prediction)
    prediction = np.round(np.real(prediction))

    # CHECK AGAINST GROUND-TRUTH
    expectation = np.array([*list(new_state), next_action])
    correctness_arr[i] = np.array_equal(prediction, expectation)

    # UPDATE
    current_state = new_state
    action = next_action

print(f"Koopman final state: {prediction}")
print(f"Ground-truth final state: {[*list(current_state), action]}")
# These are surprisingly close running a few tests!

for i in range(num_steps):
    if not correctness_arr[i]:
        print(f"Step at which Koopman prediction diverges: {i}")
        break