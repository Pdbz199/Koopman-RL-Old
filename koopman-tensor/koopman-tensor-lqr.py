#%% Imports
import importlib
import gym
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy as sp
import sys
sys.path.append("../")
import estimate_L
import auxiliaries
import algorithmsv2

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
y = np.append(x, [x[0]**2], axis=0)

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

def C2_func(x):
    np.random.seed( np.abs( int( hash(str(x)) / (10**10) ) ) )
    return np.array([
        np.random.uniform(-100,100),
        np.random.uniform(-100,100),
        np.random.uniform(-100,100)
    ])

def U_builder(X):
    U = []
    for x in X.T:
        U.append([-C2_func(x)@x])
    return np.array(U).T

#%% Randomly controlled system (currently takes 5-ever to run
# X = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2_func(x)[:2] @ x) - (C2_func(x)[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
# X = X.y
# Y = np.roll(X, -1, axis=1)[:,:-1]
# X = X[:,:-1]
# U = U_builder(X)

#%% Standard LQR controlled system
X_opt = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2[:2] @ x) - (C2[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
X_opt = X_opt.y
Y_opt = np.roll(X_opt, -1, axis=1)[:,:-1]
X_opt = X_opt[:,:-1]
U_opt = (-C@X_opt).reshape(1,-1)

#%% Koopman LQR controlled system
# X2 = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2[:2] @ x) - (C2[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
# X2 = X2.y[:,:-2]
# U2 = getKoopmanAction(X2).reshape(1,-1)
# Y2 = np.apply_along_axis(lambda x: np.append(x, [x[0]**2]), axis=0, arr=X)

def cost(x, u):
    return (x @ Q @ x + u * R * u)

#%% Matrix builder functions
def phi(x):
    return np.array([1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]])

def psi(u):
    return np.array([float(1), float(u), float(u**2)])

def getPhiMatrix(X):
    Phi_X = []
    for x in X.T:
        Phi_X.append(phi(x))

    return np.array(Phi_X).T

def getPsiMatrix(U):
    Psi_U = []
    for u in U.T:
        Psi_U.append(psi(u))

    return np.array(Psi_U).T

#%%
Phi_X = getPhiMatrix(X_opt)
Psi_U = getPsiMatrix(U_opt)
print("Phi_X shape:", Phi_X.shape)
print("Psi_U shape:", Psi_U.shape)

num_lifted_state_observations = Phi_X.shape[1]
num_lifted_state_features = Phi_X.shape[0]
num_lifted_action_observations = Psi_U.shape[1]
num_lifted_action_features = Psi_U.shape[0]

# @nb.njit(fastmath=True)
def getPsiPhiMatrix(Psi_U, Phi_X):
    psiPhiMatrix = np.empty((num_lifted_action_features * num_lifted_state_features, num_lifted_state_observations))

    for i in range(num_lifted_state_observations):
        kron = np.kron(Psi_U[:,i], Phi_X[:,i])
        psiPhiMatrix[:,i] = kron

    return psiPhiMatrix

#%%
psiPhiMatrix = getPsiPhiMatrix(Psi_U, Phi_X)
print("PsiPhiMatrix shape:", psiPhiMatrix.shape)
M = estimate_L.ols(psiPhiMatrix.T, getPhiMatrix(Y_opt).T).T
print("M shape:", M.shape)
assert M.shape == (num_lifted_state_features, num_lifted_state_features * num_lifted_action_features)

K = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M.shape[0]):
    K[i] = M[i].reshape((num_lifted_state_features, num_lifted_action_features), order='F')
print("K shape:", K.shape)

def K_u(u):
    return np.einsum('ijz,z->ij', K, psi(u))

print("Psi U[0,0]:", psi(U_opt[0,0]))
print("K_u shape:", K_u(U_opt[0,0]).shape)

#%%
def l2_norm(true_state, predicted_state):
    return np.sum( np.power( ( true_state - predicted_state ), 2 ) )

#%% Training error
norms = []
starting_point = 0
for i in range(Y_opt.shape[1]):
    actual_phi_x_prime = phi(Y_opt[:,starting_point+i])
    predicted_phi_x_prime = K_u(U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])

    norms.append(l2_norm(actual_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)
# print(norms)
print("Mean training norm:", norms.mean())

#%% Single-step prediction error with optimal controller
norms = []
starting_point = 0
for i in range(Y_opt.shape[1]):
    actual_phi_x_prime = phi(Y_opt[:,starting_point+i])
    predicted_phi_x_prime = K_u(U_opt[0,starting_point+i]) @ phi(X_opt[:,starting_point+i])

    norms.append(l2_norm(actual_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)
print("Mean single-step prediction norm:", norms.mean())

#%%