# creating maybe an MDP with, say,
# 5 states, and 3 actions, with deterministic transition

"""
state0 - action0 = state1
state0 - action1 = state2
state0 - action2 = state3

state1 - action0 = state0
state1 - action1 = state2
state1 - action2 = state3

state2 - action0 = state1
state2 - action1 = state3
state2 - action2 = state4

state3 - action0 = state1
state3 - action1 = state2
state3 - action2 = state4

state4 - action0 = state0
state4 - action1 = state2
state4 - action2 = state3
"""

#%%
import numpy as np
import sys
sys.path.append('../')
import estimate_L

def f(x, u):
    if x == 0:
        if u == 0:
            return 1
        if u == 1:
            return 2
        if u == 2:
            return 3
    if x == 1:
        if u == 0:
            return 0
        if u == 1:
            return 2
        if u == 2:
            return 3
    if x == 2:
        if u == 0:
            return 1
        if u == 1:
            return 3
        if u == 2:
            return 4
    if x == 3:
        if u == 0:
            return 1
        if u == 1:
            return 2
        if u == 2:
            return 4
    if x == 4:
        if u == 0:
            return 0
        if u == 1:
            return 2
        if u == 2:
            return 3

def psi(u):
    psi_u = np.zeros((3,1))
    psi_u[u] = 1
    return psi_u

def phi(x):
    phi_x = np.zeros((5,1))
    phi_x[x] = 1
    return phi_x

#%% Define X, Y, and U
N = 1000
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
    Phi_X[:,i] = phi(int(x[0]))[:,0]

Phi_Y = np.empty((d_phi,N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(int(y[0]))[:,0]

Psi_U = np.empty((d_psi,N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(int(u[0]))[:,0]

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M =  estimate_L.ols(kronMatrix.T, Phi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% Training error
def l2_norm(true_state, predicted_state):
    return np.sum( np.power( ( true_state - predicted_state ), 2 ) )

norms = []
for i in range(N):
    true_phi_x_prime = Phi_Y[:,i]
    predicted_phi_x_prime = K_u(K, int(U[0,i])) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())

# %%