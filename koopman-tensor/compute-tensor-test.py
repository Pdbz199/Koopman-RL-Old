# %% 
import numpy as np
import sys 
sys.path.append('../')
import estimate_L

XT = np.array(
    [[1, 0, 0, 2, 0, 2],
    [2, 0, 0, 4, 0, 4],
    [0, 0.5, 1, 0, 1, 0],
    [0, 1, 2, 0, 2, 0]]
)

YT = np.array(
    [
    [0.5, 1, 2, 1, 2, 1],
    [1, 2, 4, 2, 4, 2]
    ]
)

print(np.array_equal((np.linalg.pinv(XT@YT.T)).T, np.linalg.pinv((XT@YT.T).T)))
# %% Test
Bhat_0 = np.linalg.pinv(XT@XT.T)@XT@YT.T

Bhat_1 = estimate_L.ols(XT.T, YT.T)

np.array_equal(Bhat_0, Bhat_1)

# %% Create large sample
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

N = 1000
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
    X[:, i+1] = f(X[:,i], U[0,i])
    kronMatrix[:, i] = np.kron(X[:, i], Psi[:, i])

Phi_prime = np.roll(X, -1, axis=1)[:,:-1]
M =  estimate_L.ols(kronMatrix.T, Phi_prime.T).T

K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    K[i] = M[i].reshape((d_phi,d_psi), order = 'F')




# %%
