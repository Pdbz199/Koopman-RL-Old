#%%
import numpy as np
np.random.seed(123)
import sys
sys.path.append('../../')
import estimate_L
import observables

A = np.array([
    [1.5, 0],
    [0, 0.1]
])
B = np.array([
    [1],
    [0]
])

def F(x,u):
    return A @ x + B @ [u]

def random_control(x):
    ''' compute random control variable u '''
    u = np.random.uniform(high=10.0, low=-10.0)
    return u

def nonrandom_control(x):
    ''' compute state-dependent control variable u '''
    u = -1*x[0] #+ np.random.randn()
    return u

# def phi(x):
#     ''' Identity for DMD '''
#     return x

# def psi(u):
#     ''' Identity for DMD '''
#     return u

monomials = observables.monomials(2)

def phi(x):
    ''' Quadratic dictionary '''
    if len(x.shape) == 1:
        return monomials(x.reshape(-1,1))[:,0]
    return monomials(x)[:,0]

def psi(u):
    ''' Quadratic dictionary '''
    return phi(u)

#%% simulate system to generate data matrices
m = 500 # number of sample steps from the system.
n = 2 # dimensionality of state space
q = 1 # dimensionality of control space

#%% Control snapshotting
X = np.random.normal(loc=8, scale=5, size=(n, m*2-1))
# X = np.full((n, m*2-1), np.array([[4],[7]]))
# X[:,0] = np.array([6,16])
Y = np.empty((n, m*2-1))
U = np.empty((q, m*2-1))
# sys = UnstableSystem1(x0)
for k in range(m*2-1):
    # separate state and action
    u_k = random_control(X[:, k])
    U[:, k] = u_k
    y = F(X[:, k], u_k)
    Y[:, k] = np.squeeze(y)

#%% Build Phi and Psi matrices
d_phi = 6
d_psi = 3
N = m*2-1

Phi_X = np.empty((d_phi, N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(x)

Phi_Y = np.empty((d_phi, N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(y)

Psi_U = np.empty((d_psi, N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(u)
    
XU = np.append(X, U, axis=0)
d_phi_xu = phi(XU[:,0]).shape[0]
Phi_XU = np.empty((d_phi_xu, N))
for i,xu in enumerate(XU.T):
    Phi_XU[:,i] = phi(xu)

#%% Concatenated state and action Koopman operator
Koopman_operator = estimate_L.ols(Phi_XU[:,:-1].T, Phi_XU[:,1:].T).T

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
    return np.einsum('ijz,z->ij', K, psi(u))

#%% Training error
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

concatenated_norms = []
norms = []
for i in range(N-1):
    true_phi_xu_prime = Phi_XU[:,i+1]
    predicted_phi_xu_prime = Koopman_operator @ Phi_XU[:,i]
    concatenated_norms.append(l2_norm(true_phi_xu_prime, predicted_phi_xu_prime))

    true_phi_x_prime = Phi_Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
concatenated_norms = np.array(concatenated_norms)
norms = np.array(norms)

print("Concatenated mean norm on training data:", concatenated_norms.mean())
print("Tensor mean norm on training data:", norms.mean())





#%% State snapshotting
snapshots = np.empty((n, m))
snapshots[:, 0] = np.array([16,10])

#%% Control snapshotting
U = np.empty((q, m-1))
# sys = UnstableSystem1(x0)
for k in range(m-1):
    u_k = nonrandom_control(snapshots[:, k])
    y = F(snapshots[:, k], u_k)
    snapshots[:, k+1] = np.squeeze(y)
    U[:, k] = u_k

X = snapshots[:, :m-1]
Y = snapshots[:, 1:m]
X_concatenated = np.append(X, U, axis=0)

#%% Build Phi and Psi matrices
d_phi = 6
d_psi = 3
N = m-1

Phi_X = np.empty((d_phi, N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(x)

Phi_Y = np.empty((d_phi, N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(y)

Psi_U = np.empty((d_psi, N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(u)

XU = np.append(X, U, axis=0)
d_phi_xu = phi(XU[:,0]).shape[0]
Phi_XU = np.empty((d_phi_xu, N))
for i,xu in enumerate(XU.T):
    Phi_XU[:,i] = phi(xu)

#%% Prediction error
concatenated_norms = []
norms = []
for i in range(N-1):
    true_phi_xu_prime = Phi_XU[:,i+1]
    predicted_phi_xu_prime = Koopman_operator @ Phi_XU[:,i]
    concatenated_norms.append(l2_norm(true_phi_xu_prime, predicted_phi_xu_prime))

    true_phi_x_prime = Phi_Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
concatenated_norms = np.array(concatenated_norms)
norms = np.array(norms)

print("Concatenated mean norm on prediction data:", concatenated_norms.mean())
print("Tensor mean norm on prediction data:", norms.mean())

#%%