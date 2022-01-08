#%% Imports
# import importlib
# import gym
import numpy as np
np.random.seed(123)
# import numba as nb
# import matplotlib.pyplot as plt
# import scipy as sp
import sys
sys.path.append("../../")
# import algorithmsv2
# import auxiliaries
import estimate_L
import observables

from control import lqr
from scipy import integrate
from sklearn.kernel_approximation import RBFSampler

#%% System variable definitions
mu = -0.1
lamb = -0.5

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

K = np.array(
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

#%%
x = np.array([
    [-5.],
    [5.]
])
x2 = np.append(x, [x[0]**2], axis=0)

#%% Standard LQR
C = lqr(A, B, Q, R)[0][0]
print("Standard LQR:", C)
# C = np.array([0,2.4142]) when lamb = 1

#%% Koopman LQR
# C = [0.0, 0.61803399, 0.23445298] when lamb = -0.5
C2 = lqr(K, B2, Q2, R)[0][0]
print("Koopman LQR:", C2)
# C = np.array([0.0, 2.4142, -1.4956])
# u = ((-C[:2] @ x) - (C[2] * x[0]**2))[0]

def vf(tau, x, u):
    returnVal = ((A @ x.reshape(-1,1)) + np.array([[0], [-lamb * x[0]**2]]) + (B @ u.reshape(-1,1)))
    return returnVal[:,0]

#%% Define system
class ContinuousDynSys():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return (-self.c[:2] @ x) - (self.c[2] * x[0]**2)
    
    def sigma(self, x):
        return 0 # was np.sqrt(2/self.beta) for Double Well

class EulerMaruyama(object):
    def __init__(self, h, nSteps):
        self.h = h
        self.nSteps = nSteps
    
    def integrate(self, s, x):
        y = np.zeros((2,self.nSteps))
        y[:,0] = x
        for i in range(1, self.nSteps):
            y[:,i] = y[:,i-1] + s.b(y[:,i-1])*self.h + s.sigma(y[:,i-1])*np.sqrt(self.h)*np.random.randn()
        return np.append(y, [y[0]**2], axis=0)

init_beta = 1
init_c = np.array([
    np.random.uniform(-100,100),
    np.random.uniform(-100,100),
    np.random.uniform(-100,100)
])
h = 1e-2
num_steps_in_system = 10000
s = ContinuousDynSys(beta=init_beta, c=init_c)
em = EulerMaruyama(h, num_steps_in_system)

#%% Generate training data
m = 1000#0 # number of data points
X = 5*np.random.rand(2,m) - 2.5
# X2 = np.append(x, [x[0]**2], axis=0)
Y = np.zeros((3,m))
U = np.zeros((3,m))
for i in range(m):
    s.c = np.array([
        np.random.uniform(-100,100),
        np.random.uniform(-100,100),
        np.random.uniform(-100,100)
    ])
    U[:,i] = s.c
    y = em.integrate(s, X[:,i])
    Y[:,i] = y[:,-1]

#%% Standard LQR controlled system
X_opt = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2[:2] @ x) - (C2[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
X_opt = X_opt.y
Y_opt = np.roll(X_opt, -1, axis=1)[:,:-1]
X_opt = X_opt[:,:-1]
U_opt = (-C@X_opt).reshape(1,-1) # should this be C2?

#%% Define cost function
def cost(x, u):
    return (x @ Q2 @ x + u * R * u) # (x @ Q @ x + u * R * u)

#%% Matrix builder functions
order = 2 # maybe 4
phi = observables.monomials(order)
psi = observables.monomials(order)

#%%
N = X.shape[1]
Phi_X = phi(X)
Psi_U = psi(U)
print("Phi_X shape:", Phi_X.shape)
print("Psi_U shape:", Psi_U.shape)

dim_phi = Phi_X.shape[0]
dim_psi = Psi_U.shape[0]
print(dim_phi)
print(dim_psi)

dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]
print(dim_phi)
print(dim_psi)

# num_lifted_state_observations = Phi_X.shape[1]
# num_lifted_state_features = Phi_X.shape[0]
# num_lifted_action_observations = Psi_U.shape[1]
# num_lifted_action_features = Psi_U.shape[0]

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%%
psiPhiMatrix = getPsiPhiMatrix(Psi_U, Phi_X)
print("PsiPhiMatrix shape:", psiPhiMatrix.shape)
M = estimate_L.ols(psiPhiMatrix.T, getPhiMatrix(Y_opt).T).T
M_2 = estimate_L.SINDy(psiPhiMatrix.T, getPhiMatrix(Y_opt).T).T
M_3 = estimate_L.rrr(psiPhiMatrix.T, getPhiMatrix(Y_opt).T).T
print("M shape:", M.shape)
assert M.shape == (num_lifted_state_features, num_lifted_state_features * num_lifted_action_features)

K = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M.shape[0]):
    K[i] = M[i].reshape((num_lifted_state_features, num_lifted_action_features), order='F')
print("K shape:", K.shape)
K_2 = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M_2.shape[0]):
    K_2[i] = M_2[i].reshape((num_lifted_state_features, num_lifted_action_features), order='F')
K_3 = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M_3.shape[0]):
    K_3[i] = M_3[i].reshape((num_lifted_state_features, num_lifted_action_features), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(u))

print("Psi U[0,0]:", psi(U_opt[0,0]))
print("K_u shape:", K_u(K, U_opt[0,0]).shape)

#%%
def l2_norm(true_state, predicted_state):
    return np.sum( np.power( ( true_state - predicted_state ), 2 ) )

#%% Training error
norms = []
norms_2 = []
norms_3 = []
starting_point = 0
for i in range(Y_opt.shape[1]):
    actual_phi_x_prime = phi(Y_opt[:,starting_point+i])
    predicted_phi_x_prime = K_u(K, U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])

    norms.append(l2_norm(actual_phi_x_prime, predicted_phi_x_prime))
    norms_2.append(l2_norm(actual_phi_x_prime, K_u(K_2, U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])))
    norms_3.append(l2_norm(actual_phi_x_prime, K_u(K_3, U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])))
norms = np.array(norms)
norms_2 = np.array(norms_2)
norms_3 = np.array(norms_3)
# print(norms)
print("Mean training norm (OLS):", norms.mean())
print("Mean training norm (SINDy):", norms_2.mean())
print("Mean training norm (RRR):", norms_3.mean())

#%% Single-step prediction error with optimal controller
norms = []
norms_2 = []
norms_3 = []
starting_point = 0
for i in range(Y_opt.shape[1]):
    actual_phi_x_prime = phi(Y_opt[:,starting_point+i])
    predicted_phi_x_prime = K_u(K, U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])

    norms.append(l2_norm(actual_phi_x_prime, predicted_phi_x_prime))
    norms_2.append(l2_norm(actual_phi_x_prime, K_u(K_2, U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])))
    norms_3.append(l2_norm(actual_phi_x_prime, K_u(K_3, U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])))
norms = np.array(norms)
norms_2 = np.array(norms_2)
norms_3 = np.array(norms_3)
print("Mean single-step prediction norm (OLS):", norms.mean())
print("Mean single-step prediction norm (SINDy):", norms_2.mean())
print("Mean single-step prediction norm (RRR):", norms_3.mean())

#%%