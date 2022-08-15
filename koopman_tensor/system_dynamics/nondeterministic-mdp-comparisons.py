#%% Imports 
import numpy as np
import sys
sys.path.append('../../')
import estimate_L

#%% Transition Tensor
P = np.array([
    [
        [0, 0, 0],
        [1/2, 1/4, 1/2],
        [1/4, 1/3, 1/3],
        [1, 1/4, 0],
        [0, 0, 1/4]
    ],
    [
        [1/3, 1/2, 1/4],
        [0, 0, 0],
        [1/4, 1/3, 0],
        [0, 1/4, 0],
        [0, 0, 1/4]
    ],
    [
        [1/3, 0, 1/2],
        [1/2, 1/4, 0],
        [0, 0, 0],
        [0, 1/2, 1],
        [1, 0, 1/4]
    ],
    [
        [1/3, 1/2, 0],
        [0, 0, 1/2],
        [1/4, 0, 1/3],
        [0, 0, 0],
        [0, 1, 1/4]
    ],
    [
        [0, 0, 1/4],
        [0, 1/2, 0],
        [1/4, 1/3, 1/3],
        [0, 0, 0],
        [0, 0, 0]
    ]
])

#%% System dynamics
d_x = 5
d_u = 3

enumerated_extended_states = []
for x in range(d_x):
    for u in range(d_u):
        enumerated_extended_states.append([x,u])
enumerated_extended_states = np.array(enumerated_extended_states)

def f(x, u):
    return np.random.choice(np.arange(5), p=P[:,int(x),int(u)])

#%% Define and build X, Y, and U
split_datasets = {}
for action in range(d_u):
    split_datasets[action] = {"x":[],"x_prime":[]}

N = 10_000
d_phi = 5
d_psi = 3
x0 = np.array([[0]])
X = np.empty((1,N+1))
X[:,0] = x0[:,0]
U = np.empty((1,N))
for i in range(N):
    U[0, i] = np.random.choice([i for i in range(d_u)])
    next_state = f(X[0,i], U[0,i])
    X[0, i+1] = next_state
    split_datasets[U[0,i]]['x'].append(X[0,i])
    split_datasets[U[0,i]]['x_prime'].append(X[0,i+1])

Y = np.roll(X, -1, axis=1)[:,:-1]
X = X[:,:-1]

XU = np.append(X, U, axis=0)

#%% Build Phi and Psi matrices
def one_hot_encoder(vector_dimension, index):
    vector = np.zeros((vector_dimension,1))
    vector[index] = 1
    return vector

def phi(x):
    return one_hot_encoder(d_x, x)

def phi_xu(xu):
    phi = np.zeros((d_x*d_u,1))
    for i,enumerated_extended_state in enumerate(enumerated_extended_states):
        if np.array_equal(enumerated_extended_state, xu):
            phi[i] = 1
            return phi
    
def psi(u):
    return one_hot_encoder(d_u, u)

Phi_X = np.empty((d_phi,N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(int(x[0]))[:,0]

Phi_Y = np.empty((d_phi,N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(int(y[0]))[:,0]

Phi_XU = np.empty((d_x*d_u,N))
for i,xu in enumerate(XU.T):
    Phi_XU[:,i] = phi_xu(xu)[:,0]

for action in split_datasets:
    split_datasets[action]['x'] = np.array(split_datasets[action]['x']).T
    split_datasets[action]['x_prime'] = np.array(split_datasets[action]['x_prime']).T

    split_datasets[action]['phi_x'] = []
    for u in split_datasets[action]['x']:
        split_datasets[action]['phi_x'].append(phi(int(u))[:,0])
    split_datasets[action]['phi_x_prime'] = []
    for u in split_datasets[action]['x_prime']:
        split_datasets[action]['phi_x_prime'].append(phi(int(u))[:,0])

    split_datasets[action]['phi_x'] = np.array(split_datasets[action]['phi_x'])
    split_datasets[action]['phi_x_prime'] = np.array(split_datasets[action]['phi_x_prime'])

Psi_U = np.empty((d_psi,N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(int(u[0]))[:,0]

#%% Koopman operator for extended state
extended_state_koopman_operator = estimate_L.ols(Phi_XU[:,:-1].T, Phi_XU[:,1:].T).T
extended_state_to_state = estimate_L.ols(Phi_XU.T, Phi_X.T).T
estimated_P = np.empty_like(P)
for enumerated_extended_state in enumerated_extended_states:
    estimated_P[:,enumerated_extended_state[0], enumerated_extended_state[1]] = \
        np.reshape((extended_state_to_state @ extended_state_koopman_operator @ phi_xu(enumerated_extended_state)), (d_phi,))

#%% Separate Koopman operator per action
separate_koopman_operators = []
for action in range(d_u):
    separate_koopman_operators.append(estimate_L.ols(
        split_datasets[action]['phi_x'], split_datasets[action]['phi_x_prime']
    ).T)
separate_koopman_operators = np.array(separate_koopman_operators)

multi_koopman_estimated_P = np.empty_like(P)
for enumerated_extended_state in enumerated_extended_states:
    x = enumerated_extended_state[0]
    u = enumerated_extended_state[1]
    multi_koopman_estimated_P[:,x,u] = separate_koopman_operators[u] @ phi(x)[:,0]

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(int(u))[:,0])

#%% Training error
def l2_norm(true_state, predicted_state):
    return np.sum( np.power( ( true_state - predicted_state ), 2 ) )

norms = []
for i in range(N):
    true_phi_x_prime = Phi_Y[:,i]
    predicted_phi_x_prime = K_u(K, U[0,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())

print("F-norm between K (result of tensor approach) and P:", l2_norm(P, K))
print("F-norm between estimated P from extended state Koopman operator and true P:", l2_norm(P, estimated_P))
print("F-norm between K (result of Tensor approach) and estimated P from extended state Koopman operator:", l2_norm(K, estimated_P))
print("F-norm between estimated P from multiple Koopman operators and true P:", l2_norm(P, multi_koopman_estimated_P))
print("F-norm between K (result of Tensor approach) and estimated P from multiple Koopman operators:", l2_norm(K, multi_koopman_estimated_P))

#%%