#%%
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
import estimate_L
import observables

#%% GENERATE TRAINING DATA FOR FIXED U

import numpy as np
import matplotlib.pyplot as plt

class DoubleWell():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return -x**3 + 2*x + self.c
    
    def sigma(self, x):
        return np.sqrt(2/self.beta)

class EulerMaruyama(object):
    def __init__(self, h, nSteps):
        self.h = h
        self.nSteps = nSteps
    
    def integrate(self, s, x):
        y = np.zeros(self.nSteps)
        y[0] = x
        for i in range(1, self.nSteps):
            y[i] = y[i-1] + s.b(y[i-1])*self.h + s.sigma(y[i-1])*np.sqrt(self.h)*np.random.randn()
        return y

#%% Create double-well system and integrator
s = DoubleWell(beta=2, c=0)
em = EulerMaruyama(1e-3, 10000)

# Starting point
x0 = 5*np.random.rand() - 2.5

#%% Generate one trajectory
# y = em.integrate(s, x0)
# plt.clf()
# plt.plot(y)

#%% Generate training data
m = 1000#0 # number of data points
X = 5*np.random.rand(1,m) - 2.5
Y = np.zeros((1,m))
U = np.zeros((1,m))
for i in range(m):
    s.c = np.random.uniform(-2.0, 2.0)
    U[0,i] = s.c
    y = em.integrate(s, x0)
    Y[0,i] = y[-1]

# plt.figure()
# plt.hist(Y, 50)

#%% Koopman Tensor
order = 10
phi = observables.monomials(order)
psi = observables.monomials(1)

#%% Build Phi and Psi matrices
Phi_X = phi(X)
Phi_Y = phi(Y)
Psi_U = psi(U)
dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]
N = X.shape[1]

print(Phi_X.shape)

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
B = estimate_L.ols(Phi_X.T, X.T)

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    # return np.einsum('ijz,kz->ij', K, psi(u))
    return np.einsum('ijz,z->ij', K, psi(u)[:,0])

#%% L2 Norm
def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        raise Exception(f'The dimensions of the parameters did not match ({true_state.shape} and {predicted_state.shape}) and therefore cannot be compared.')
    
    err = true_state - predicted_state
    return np.sum(np.power(err, 2))

#%% Training error
training_prediction_norms = np.zeros(N)
training_phi_prediction_norms = np.zeros(N)
for i in range(X.shape[1]):
    phi_x = Phi_X[:,i].reshape(-1,1)
    u = U[:,i].reshape(-1,1)

    true_x_prime = Y[:,i].reshape(-1,1)
    true_phi_x_prime = Phi_Y[:,i].reshape(-1,1)
    predicted_phi_x_prime = K_u(K, u) @ phi_x

    training_prediction_norms[i] = l2_norm(true_x_prime, B.T @ predicted_phi_x_prime)
    training_phi_prediction_norms[i] = l2_norm(true_phi_x_prime, predicted_phi_x_prime)
print("TRAINING: Mean l2 norm between true and predicted x_prime:", np.mean(training_prediction_norms))
print("TRAINING: Mean l2 norm between true and predicted phi_x_prime:", np.mean(training_phi_prediction_norms))

#%% Testing error
path_length = 1000#0
starting_x = 5*np.random.rand() - 2.5
last_x = np.array([[starting_x]])

testing_prediction_norms = np.zeros(N)
testing_phi_prediction_norms = np.zeros(N)
for i in range(path_length):
    s.c = np.random.uniform(-2.0, 2.0) # Random control

    true_x_prime = np.array([[em.integrate(s, last_x[0,0])[-1]]]) # Compute true x_prime
    true_phi_x_prime = phi(true_x_prime) # Compute phi of true x_prime

    predicted_phi_x_prime = K_u(K, np.array([[s.c]])) @ phi(last_x)
    predicted_x_prime = B.T @ predicted_phi_x_prime

    testing_prediction_norms[i] = l2_norm(true_x_prime, predicted_x_prime)
    testing_phi_prediction_norms[i] = l2_norm(true_phi_x_prime, predicted_phi_x_prime)

    last_x = true_x_prime
print("TESTING: Mean l2 norm between true and predicted x_prime:", np.mean(testing_prediction_norms))
print("TESTING: Mean l2 norm between true and predicted phi_x_prime:", np.mean(testing_phi_prediction_norms))

#%%