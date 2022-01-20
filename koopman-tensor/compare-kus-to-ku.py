#%%
import numpy as np

import sys
sys.path.append('../')
import estimate_L
import utilities

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

#%%
range_5 = np.arange(5)
def f(x, u):
    return np.random.choice(range_5, p=P[:,int(x),int(u)])

def phi(x):
    psi_u = np.zeros((5,x.shape[1]))
    psi_u[x[0].astype(int),np.arange(0,x.shape[1])] = 1
    return psi_u

def psi(u):
    psi_u = np.zeros((3,u.shape[1]))
    psi_u[u[0].astype(int),np.arange(0,u.shape[1])] = 1
    return psi_u

#%% Define X, Y, and U
N = 10000
d_phi = 5
d_psi = 3
x0 = np.array([[0]])
X = np.empty((1,N+1))
X[:,0] = x0[:,0]
U = np.empty((1,N))
for i in range(N):
    U[0, i] = np.random.choice([0,1,2])
    next_state = f(X[0,i], U[0,i])
    X[0, i+1] = next_state

Y = np.roll(X, -1, axis=1)[:,:-1]
X = X[:,:-1]

#%% Build Phi and Psi matrices
Phi_X = np.empty((d_phi,N))
for i,x in enumerate(X.T):
    x = np.vstack(x)
    Phi_X[:,i] = phi(x)[:,0]

Phi_Y = np.empty((d_phi,N))
for i,y in enumerate(Y.T):
    y = np.vstack(y)
    Phi_Y[:,i] = phi(y)[:,0]

Psi_U = np.empty((d_psi,N))
for i,u in enumerate(U.T):
    u = np.vstack(u)
    Psi_U[:,i] = psi(u)[:,0]

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M and B_
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
B = estimate_L.ols(Phi_X.T, X.T)

#%% Reshape M into K tensor
K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

#%%
def K_u(u):
    ''' Pick out Koopman operator given an action '''
    return np.einsum('ijz,zk->ij', K, psi(u))

def K_us(us):
    ''' Pick out Koopman operator given a matrix of action column vectors '''
    return np.einsum('ijz,zk->kij', K, psi(us))

#%%
num_operators = 1000
KUs = K_us(U[:,:num_operators])
KUs_arr = np.empty((num_operators,d_phi,d_phi))
for i in range(num_operators):
    KUs_arr[i] = K_u(np.vstack(U[:,i]))

#%%
assert np.array_equal(KUs, KUs_arr)

#%%