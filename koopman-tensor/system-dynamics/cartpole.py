#%% Imports
import gym
import numpy as np
np.random.seed(123)
import time

import sys
sys.path.append('../../')
import estimate_L
import observables

def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        # print("Shape 1:", true_state.shape)
        # print("Shape 2:", predicted_state.shape)
        raise Exception(f'The dimensions of the parameters did not match ({true_state.shape} and {predicted_state.shape}) and therefore cannot be compared.')
    
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
# psi = observables.monomials(order)

# One-hot encoder
def psi(u):
    psi_u = np.zeros((env.action_space.n,u.shape[1]))
    psi_u[u[0].astype(int),np.arange(0,u.shape[1])] = 1
    return psi_u

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

print("Phi_X shape:", Phi_X.shape)
print("Psi_U shape:", Psi_U.shape)

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
num_ranks = 14

M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
M_2 = estimate_L.SINDy(kronMatrix.T, Phi_Y.T).T
M_rrrs = np.empty((num_ranks, dim_phi, dim_phi * dim_psi))
for i in range(num_ranks):
    M_rrrs[i] = estimate_L.rrr(kronMatrix.T, Phi_Y.T, rank=i+1).T
print("M shape:", M.shape)
assert M.shape == (dim_phi, dim_phi * dim_psi)

B = estimate_L.ols(Phi_X.T, X.T)
assert B.shape == (dim_phi, X.shape[0])

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')
K_2 = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K_2[i] = M_2[i].reshape((dim_phi,dim_psi), order='F')

K_rrrs = np.empty((num_ranks, dim_phi, dim_phi, dim_psi))
for i in range(num_ranks):
    K_rrr = np.empty((dim_phi, dim_phi, dim_psi))
    for j in range(dim_phi):
        K_rrr[j] = M_rrrs[i,j].reshape((dim_phi,dim_psi), order='F')
    K_rrrs[i] = K_rrr

def K_u(K, u):
    if len(u.shape) == 1:
        u = u.reshape(-1,1) # assume transposing row vector into column vector
    # u must be column vector
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% Difference error
print("Difference between K tensor and separate Koopman operators:")
for action in range(env.action_space.n):
    print(l2_norm(K_u(K, np.array([[action]])), Koopman_operators[action]))

#%% Training error
multi_norms = np.empty((N))
extended_norms = np.empty((N))
tensor_norms = np.empty((N))
tensor_norms_2 = np.empty((N))
tensor_norms_rrr = np.empty((num_ranks, N))
for i in range(N):
    x = X[:,i] # current state
    phi_x = Phi_X[:,i].reshape(-1,1) # current (lifted) state

    action = env.action_space.sample() # sample random action

    # Predictions
    multi_koopman_prediction = B.T @ (Koopman_operators[action]) @ phi_x
    extended_prediction = extended_B.T @ extended_koopman_operator @ phi(np.append(x,U[:,i]).reshape(-1,1))
    tensor_prediction = B.T @ K_u(K, np.array([[action]])) @ phi_x
    tensor_prediction_2 = B.T @ K_u(K_2, np.array([[action]])) @ phi_x

    tensor_predictions_rrr = np.empty((num_ranks, dim_x, 1))
    for j in range(num_ranks):
        tensor_predictions_rrr[j] = B.T @ K_u(K_rrrs[j], np.array([[action]])) @ phi_x

    true_x_prime = Y[:,i] # next state

    # Compute norms
    multi_norms[i] = l2_norm(true_x_prime, multi_koopman_prediction[:,0])
    extended_norms[i] = l2_norm(true_x_prime, extended_prediction[:-1,0])
    tensor_norms[i] = l2_norm(true_x_prime, tensor_prediction[:,0])
    tensor_norms_2[i] = l2_norm(true_x_prime, tensor_prediction_2[:,0])
    for j in range(num_ranks):
        tensor_norms_rrr[j,i] = l2_norm(true_x_prime, tensor_predictions_rrr[j,:,0])
print("Multi Koopman mean training error:", np.mean(multi_norms))
print("Extended mean training error:", np.mean(extended_norms))
print("Tensor mean training error (OLS):", np.mean(tensor_norms))
print("Tensor mean training error (SINDy):", np.mean(tensor_norms_2))
for i in range(num_ranks):
    print(f"Tensor mean training error (RRR, rank={i+1}):", np.mean(tensor_norms_rrr[i]))

#%% Run environment for prediction error
episodes = 10
multi_norms = []
extended_norms = []
tensor_norms = []
tensor_norms_2 = []
tensor_norms_rrr = []
for episode in range(episodes):
    observation = env.reset()
    done = False
    while not done:
        # env.render()
        phi_x = phi(observation.reshape(-1,1)) # phi applied to current state
        action = env.action_space.sample() # sample random action

        # Predictions
        multi_koopman_prediction = B.T @ (Koopman_operators[action]) @ phi_x
        extended_prediction = extended_B.T @ extended_koopman_operator @ phi(np.append(observation,action).reshape(-1,1))
        tensor_prediction = B.T @ K_u(K, np.array([[action]])) @ phi_x
        tensor_prediction_2 = B.T @ K_u(K_2, np.array([[action]])) @ phi_x
        tensor_predictions_rrr = []
        for i in range(num_ranks):
            tensor_predictions_rrr.append(B.T @ K_u(K_rrrs[i], np.array([[action]])) @ phi_x)
        print(nptensor_predictions_rrr)

        # Take one step forward in environment
        observation, reward, done, _ = env.step(action)

        # Compute norms
        multi_norms.append(l2_norm(observation, multi_koopman_prediction[:,0]))
        extended_norms.append(l2_norm(observation, extended_prediction[:-1,0]))
        tensor_norms.append(l2_norm(observation, tensor_prediction[:,0]))
        tensor_norms_2.append(l2_norm(observation, tensor_prediction_2[:,0]))
        for i in range(num_ranks):
            tensor_norms_rrr.append(l2_norm(observation, tensor_predictions_rrr[i,:,0]))
env.close()
print("Multi Koopman mean prediction error:", np.mean(multi_norms))
print("Extended mean prediction error:", np.mean(extended_norms))
print("Tensor mean training error (OLS):", np.mean(tensor_norms))
print("Tensor mean training error (SINDy):", np.mean(tensor_norms_2))
for i in range(num_ranks):
    print(f"Tensor mean training error (RRR, rank={i+1}):", np.mean(tensor_norms_rrr[i]))

#%%
