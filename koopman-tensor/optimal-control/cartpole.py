#%% Imports
import gym
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import time

import sys
sys.path.append('../../')
import algorithmsv2
import cartpole_reward
import estimate_L
# import tf_observables as observables
import observables
import tf_algorithmsv2

def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        print("Shape 1:", true_state.shape)
        print("Shape 2:", predicted_state.shape)
        raise Exception('The dimensions of the parameters did not match and therefore cannot be compared.')
    
    err = true_state - predicted_state
    return np.sum(np.power(err, 2))

#%% Load environment
env = gym.make('CartPole-v0')

#%% Load data
X_0 = np.load('../../random-agent/cartpole-states-0.npy').T
X_1 = np.load('../../random-agent/cartpole-states-1.npy').T
Y_0 = np.load('../../random-agent/cartpole-next-states-0.npy').T
Y_1 = np.load('../../random-agent/cartpole-next-states-1.npy').T
X_data = { 0: X_0, 1: X_1 }
Y_data = { 0: Y_0, 1: Y_1 }

X = np.append(X_data[0], X_data[1], axis=1)
Y = np.append(Y_data[0], Y_data[1], axis=1)
U = np.empty([1,X.shape[1]])
for i in range(X_data[0].shape[1]):
    U[:,i] = [0]
for i in range(X_data[1].shape[1]):
    U[:,i+X_data[0].shape[1]] = [1]
XU = np.append(X, U, axis=0) # extended states

dim_x = X.shape[0] # dimension of each data point (snapshot)
dim_u = U.shape[0] # dimension of each action
N = X.shape[1] # number of data points (snapshots)

#%% Matrix builder functions
order = 2
phi = observables.monomials(order)
psi = observables.monomials(order)

# One-hot encoder
# def psi(u):
#     psi_u = np.zeros((env.action_space.n,1))
#     psi_u[int(u),0] = 1
#     return psi_u

#%% Compute Phi and Psi matrices + dimensions
Phi_X = phi(X)
Phi_Y = phi(Y)

Phi_XU = phi(XU)

Psi_U = psi(U)
# Psi_U = np.empty([env.action_space.n,N])
# for i,u in enumerate(U.T):
#     Psi_U[:,i] = psi(u[0])[:,0]

dim_phi = Phi_X.shape[0]
dim_psi = Psi_U.shape[0]

# print("Phi_X shape:", Phi_X.shape)
# print("Psi_U shape:", Psi_U.shape)

#%% Estimate Koopman operator for each action
Koopman_operators = np.empty((env.action_space.n,dim_phi,dim_phi))
for action in range(env.action_space.n):
    if np.array_equal(phi(X_data[action]), [1]):
        Koopman_operators[action] = np.zeros([dim_phi,dim_phi])
        continue
    Koopman_operators[action] = estimate_L.ols(phi(X_data[action]).T, phi(Y_data[action]).T).T
Koopman_operators = np.array(Koopman_operators)

#%% Estimate extended state Koopman operator
extended_koopman_operator = estimate_L.ols(Phi_XU[:,:-1].T, Phi_XU[:,1:].T).T
extended_B = estimate_L.ols(Phi_XU.T, XU.T)

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M and B matrices
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
# print("M shape:", M.shape)
assert M.shape == (dim_phi, dim_phi * dim_psi)

B = estimate_L.ols(Phi_X.T, X.T)
assert B.shape == (dim_phi, X.shape[0])

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    if len(u.shape) == 1:
        u = u.reshape(-1,1) # assume transposing row vector into column vector
    # u must be column vector
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% Control setup
import matplotlib.pyplot as plt

All_U = np.array([[0,1]])
u_bounds = np.array([0,1])

def cost(x,u):
    return -cartpole_reward.defaultCartpoleReward(x,u)

#%% Control
# algos = algorithmsv2.algos(X, All_U, u_bounds[0], u_bounds[1], phi, psi, K, cost, epsilon=1, bellmanErrorType=0)
# bellmanErrors, gradientNorms = algos.algorithm2()
algos = tf_algorithmsv2.Algorithms(X, All_U, phi, psi, K, cost)
algos.algorithm2()

#%% Plots
# plt.plot(np.arange(len(bellmanErrors)), bellmanErrors)
# plt.show()
# plt.plot(np.arange(len(gradientNorms)), gradientNorms)
# plt.show()

#%%
print(algos.w)

# print(algos.pi(tf.stack([[0]]), tf.reshape(X[:,1511], [X[:,0].shape[0],1]), algos.w))
# print(algos.pi(tf.stack([[1]]), tf.reshape(X[:,1511], [X[:,0].shape[0],1]), algos.w))

#%% Run policy in environment
# episodes = 100
# multi_norms = []
# extended_norms = []
# tensor_norms = []
# for episode in range(episodes):
#     observation = env.reset()
#     done = False
#     while not done:
#         env.render()
#         phi_x = phi(observation.reshape(-1,1)) # phi applied to current state
#         reshapenObservation = tf.reshape(observation, [X[:,0].shape[0],1])
#         p0 = algos.pi_u(tf.stack([[0]]), reshapenObservation)
#         p = [p0, 1-p0]
#         action = np.random.choice([0,1], p=p)

#         # Take one step forward in environment
#         observation, reward, done, _ = env.step(action)
# env.close()

#%%