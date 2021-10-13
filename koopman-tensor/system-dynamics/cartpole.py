#%% Imports
import gym
import numpy as np

import sys
sys.path.append('../../')
import estimate_L
import observables

def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        raise Exception('The dimensions of the parameters did not match and therefore cannot be compared.')

    return np.sum(
        np.power((true_state - predicted_state), 2)
    )

#%% Load environment
env = gym.make('CartPole-v0')

#%% Load data
X = np.load('../../random-agent/cartpole-states.npy').T
Y = np.roll(X, -1, axis=1)[:,:-1]
X = X[:,:-1]
U = np.load('../../random-agent/cartpole-actions.npy').reshape(1,-1)[:,:-1]

XU = np.append(X, U, axis=0)

N = X.shape[1]

X_0 = np.load('../../random-agent/cartpole-states-0.npy').T
Y_0 = np.load('../../random-agent/cartpole-next-states-0.npy').T
X_1 = np.load('../../random-agent/cartpole-states-1.npy').T
Y_1 = np.load('../../random-agent/cartpole-next-states-1.npy').T

num_unique_actions = 2

#%% Matrix builder functions
# def phi(x):
#     return x

def psi(u):
    psi_u = np.zeros((num_unique_actions,1))
    psi_u[u,0] = 1
    return psi_u

order = 2
phi = observables.monomials(order)
# psi = observables.monomials(order)

#%% Compute Phi and Psi matrices + dimensions
Phi_X = phi(X)
Phi_Y = phi(Y)
Phi_X_0 = phi(X_0)
Phi_X_1 = phi(X_1)
Phi_Y_0 = phi(Y_0)
Phi_Y_1 = phi(Y_1)
Phi_XU = phi(XU)
# Psi_U = psi(U)
Psi_U = np.empty([num_unique_actions,N])
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(u[0])[:,0]
dim_phi = Phi_X.shape[0]
dim_psi = Psi_U.shape[0]
print("Phi_X shape:", Phi_X.shape)
print("Psi_U shape:", Psi_U.shape)

#%% Estimate Koopman operator for each action
K_0 = estimate_L.ols(Phi_X_0.T, Phi_Y_0.T).T
K_1 = estimate_L.ols(Phi_X_1.T, Phi_Y_1.T).T

#%% Estimate extended state Koopman operator
extended_koopman_operator = estimate_L.ols(Phi_XU[:,:-1].T, Phi_XU[:,1:].T).T
extended_B = estimate_L.ols(Phi_XU.T, XU.T)

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])
# kronMatrix = np.kron(Psi_U, Phi_X)[:,::N+1]

#%% Estimate M and B matrices
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
print("M shape:", M.shape)
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

#%% Run environment
episodes = 1000
multi_norms = []
extended_norms = []
tensor_norms = []
for episode in range(episodes):
    observation = env.reset()
    done = False
    while not done:
        # env.render()
        phi_x = phi(observation.reshape(-1,1))
        action = env.action_space.sample() # sample random action

        # Predictions
        multi_koopman_prediction = B.T @ (K_0 if action == 0.0 else K_1) @ phi_x
        extended_prediction = extended_B.T @ extended_koopman_operator @ phi(np.append(observation,action).reshape(-1,1))
        tensor_prediction = B.T @ K_u(K, np.array([[action]])) @ phi_x

        # Take one step forward in environment
        observation, reward, done, _ = env.step(action)

        # Compute norms
        multi_norms.append(l2_norm(observation, multi_koopman_prediction[:,0]))
        extended_norms.append(l2_norm(observation, extended_prediction[:-1,0]))
        tensor_norms.append(l2_norm(observation, tensor_prediction[:,0]))
env.close()

multi_norms = np.array(multi_norms)
extended_norms = np.array(extended_norms)
tensor_norms = np.array(tensor_norms)
print("Multi Koopman mean prediction error:", np.mean(multi_norms))
print("Extended mean prediction error:", np.mean(extended_norms))
print("Tensor mean prediction error:", np.mean(tensor_norms))

#%%
