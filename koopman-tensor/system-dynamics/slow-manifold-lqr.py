import numpy as np
np.random.seed(123)
from scipy import integrate

import sys
sys.path.append("../../")
import estimate_L
import observables

#%% System variable definitions
mu = -0.1
lamb = -0.5

# Non-linear
A = np.array([
    [mu, 0   ],
    [0,  lamb]
])
B = np.array([
    [0],
    [1]
])
Q = np.identity(2)
R = 1

# These variables have already been liften into a space in which they are linear
A2 = np.array(
    [[mu, 0,    0    ],
     [0,  lamb, -lamb],
     [0,  0,    2*mu ]]
)
B2 = np.array(
    [[0],
     [1],
     [0]]
)
Q2 = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
)
# R = 1

#%% System dynamics
x = np.array([
    [-5.],
    [5.]
])
y = np.append(x, [x[0]**2], axis=0)

X = []
U = []

def vf(tau, x, u):
    X.append(x)
    U.append(u)

    returnVal = ((A @ x.reshape(-1,1)) + np.array([[0], [-lamb * x[0]**2]]) + (B @ u.reshape(-1,1)))
    return returnVal[:,0]

def C2_func():
    return np.random.uniform(-100,100,size=3)

#%% Randomly controlled system (currently takes 5-ever to run)
# _X = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2_func()[:2] @ x) - (C2_func()[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
# X = X.y
# Y = np.roll(_X, -1, axis=1)[:,:-1]

#%% Simpler dynamics
def F(x, u):
    return A@x + B@u

N = 1000
X = np.random.uniform(-500,500,size=(2,N)) # Random starting points
U = np.random.uniform(-100,100,size=(1,N)) # Random actions
Y = F(X, U) # Move one step forward

#%% Koopman Tensor
order = 2 # Does really poorly when order = 4
phi = observables.monomials(order)
psi = observables.monomials(order)

#%% Compute Phi and Psi matrices + dimensions
Phi_X = phi(X)
Phi_Y = phi(Y)
Psi_U = psi(U)

dim_phi = Phi_X.shape[0]
dim_psi = Psi_U.shape[0]

print("Phi_X shape:", Phi_X.shape)
print("Psi_U shape:", Psi_U.shape)

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M and B matrices
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
_B = estimate_L.ols(Phi_X.T, X.T)

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    if len(u.shape) == 1:
        u = u.reshape(-1,1) # assume transposing row vector into column vector
    # u must be column vector
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% L2 Norm
def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        # print("Shape 1:", true_state.shape)
        # print("Shape 2:", predicted_state.shape)
        raise Exception(f'The dimensions of the parameters did not match ({true_state.shape} and {predicted_state.shape}) and therefore cannot be compared.')
    
    err = true_state - predicted_state
    return np.sum(np.power(err, 2))

#%% Training error
tensor_norms = np.zeros((N))
for i in range(N):
    phi_x = Phi_X[:,i].reshape(-1,1)
    u = U[:,i].reshape(-1,1)

    actual_x_prime = Y[:,i].reshape(-1,1)
    predicted_x_prime = _B.T @ K_u(K, u) @ phi_x

    tensor_norms[i] = l2_norm(actual_x_prime, predicted_x_prime)
print("Tensor training error mean norm:", np.mean(tensor_norms))

#%% Prediction error
curr_x = np.random.uniform(-500,500,size=(2,1)) # Random starting point
curr_u = np.random.uniform(-100,100,size=(1,1)) # Random starting action
curr_y = F(curr_x, curr_u)

tensor_norms = np.zeros((N))
for i in range(N):
    actual_x_prime = F(curr_x, curr_u)
    predicted_x_prime = _B.T @ K_u(K, curr_u) @ phi(curr_x)

    tensor_norms[i] = l2_norm(actual_x_prime, predicted_x_prime)

    curr_x = actual_x_prime
    curr_u = np.random.uniform(-100,100,size=(1,1))
print("Tensor prediction error mean norm:", np.mean(tensor_norms))

#%%