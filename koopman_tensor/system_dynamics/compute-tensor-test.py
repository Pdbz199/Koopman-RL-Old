#%%
import numpy as np
import sys 
sys.path.append('../../')
import estimate_L

XT = np.array(
    [[1, 0, 0, 2, 0, 2],
    [2, 0, 0, 4, 0, 4],
    [0, 0.5, 1, 0, 1, 0],
    [0, 1, 2, 0, 2, 0]])

YT = np.array(
    [
    [0.5, 1, 2, 1, 2, 1],
    [1, 2, 4, 2, 4, 2]
    ]
)

print(np.array_equal((np.linalg.pinv(XT@YT.T)).T, np.linalg.pinv((XT@YT.T).T)))
#%% Test
Bhat_0 = np.linalg.pinv(XT@XT.T)@XT@YT.T

Bhat_1 = estimate_L.ols(XT.T, YT.T)

np.array_equal(Bhat_0, Bhat_1)

# Can we try multiple initial conditions?
#%% Create large sample
def f(x, u):
    if u == 0:
        return x/2
    return x*2

def psi(u):
    if u == 0:
        return np.array([[1], [0]])
    return np.array([[0], [1]])

def phi(x):
    return x

N = 10
d_phi = 2
d_psi = 2
x0 = np.array([[2],[4]])
X = np.empty((x0.shape[0],N+1))
X[:, 0] = x0[:,0]
U = np.empty((1,N))
Psi = np.empty((d_psi, N))
kronMatrix = np.empty((d_psi*d_phi, N))
for i in range(N):
    U[0, i] = np.round(np.random.uniform(0,1))
    Psi[:, i] = psi(U[0,i])[:, 0]
    next_state = f(X[:,i], U[0,i])
    X[:, i+1] = next_state
    kronMatrix[:, i] = np.kron(Psi[:, i], X[:, i])

state = np.array([20,40])
X_0 = [state]
X_1 = [state]
krons_0 = [np.kron(psi(0)[:,0],state)]
krons_1 = [np.kron(psi(1)[:,0],state)]
for i in range(N):
    X_0.append(X_0[-1]/2)
    krons_0.append(np.kron(psi(0)[:,0],X_0[-1]))
    X_1.append(X_1[-1]*2)
    krons_1.append(np.kron(psi(1)[:,0],X_1[-1]))

X_0 = np.array(X_0).T
X_1 = np.array(X_1).T
Y_0 = np.roll(X_0, -1, axis=1)[:,:-1]
Y_1 = np.roll(X_1, -1, axis=1)[:,:-1]
X_0 = X_0[:,:-1]
X_1 = X_1[:,:-1]
krons_0 = np.array(krons_0).T[:,:-1]
krons_1 = np.array(krons_1).T[:,:-1]

print("Dataset rank: ", np.linalg.matrix_rank(X_0))

K_0 = estimate_L.ols(X_0.T, Y_0.T).T
K_1 = estimate_L.ols(X_1.T, Y_1.T).T

K_kron_0 = estimate_L.ols(krons_0.T, Y_0.T).T
K_kron_1 = estimate_L.ols(krons_1.T, Y_1.T).T

Phi_prime = np.roll(X, -1, axis=1)[:,:-1]
M =  estimate_L.ols(kronMatrix.T, Phi_prime.T).T

K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%%
