#%% KLUS'S IMPLEMENTATAION

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import domain
import observables
# import matplotlib.pyplot as plt

def sortEig(A, evs=5, which='LM'):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.
    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])

def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

def gedmd(X, Y, Z, psi, evs=5, operator='K'):
    '''
    Generator EDMD for the Koopman operator. The matrices X and Y
    contain the input data. For stochastic systems, Z contains the
    diffusion term evaluated in all data points X. If the system is
    deterministic, set Z = None.
    '''
    PsiX = psi(X)
    dPsiY = np.einsum('ijk,jk->ik', psi.diff(X), Y)
    if not (Z is None): # stochastic dynamical system
        n = PsiX.shape[0] # number of basis functions
        ddPsiX = psi.ddiff(X) # second-order derivatives
        S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
        for i in range(n):
            dPsiY[i, :] += 0.5*np.sum( ddPsiX[i, :, :, :] * S, axis=(0,1) )

    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ dPsiY.T
    if operator == 'P': C_1 = C_1.T

    A = sp.linalg.pinv(C_0) @ C_1

    d, V = sortEig(A, evs, which='SM')

    norms = []
    for i in range(PsiX.shape[1]): # PsiX.shape = (rows,columns)
        true_phi_x_prime = dPsiY[:,i]
        predicted_phi_x_prime = A @ PsiX[:,i]
        norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
    norms = np.array(norms)
    print("Mean norm on training data from gedmd:", norms.mean())
    # Mean norm on training data from gedmd: 2099137637571.601

    return (A, d, V)

# define domain
bounds = np.array([[-2, 2], [-1.5, 1.5]])
boxes = np.array([20, 15])
Omega = domain.discretization(bounds, boxes)

# define system
def b(x):
     return np.vstack((-4*x[0, :]**3 + 4*x[0, :], -2*x[1, :]))
 
def sigma(x):
    n = x.shape[1]
    y = np.zeros((2, 2, n))
    y[0, 0, :] = 0.7
    y[0, 1, :] = x[0, :]
    y[1, 1, :] = 0.5
    return y

# define observables
order = 10
phi = observables.monomials(order)
def psi(u):
    return 1

# generate data
X = Omega.randPerBox(100)
Y = b(X)
Z = sigma(X)

def printVector(x, name = None, k = 8):
    '''Prints the vector like Matlab.'''
    n = x.size
    c = 0
    if name != None: print(name + ' = ')
    while c < n:
        print('\033[94m  (columns %s through %s)\033[0m' % (c, min(c+k, n)-1))
        for j in range(c, min(c+k, n)):
            print('  % 10.5f' % x[j], end = '')
        print('')
        c += k

# apply generator EDMD
evs = 3 # number of eigenvalues/eigenfunctions to be computed
K, d, V = gedmd(X, Y, Z, phi, evs=evs, operator='K')
printVector(np.real(d), 'd')

# plot eigenfunctions
# c = Omega.midpointGrid()
# Psi_c = psi(c)
# for i in range(evs):
#     plt.figure(i+1)
#     plt.clf()
#     Omega.plot(np.real( V[:, i].T @ Psi_c ), mode='3D')


#%% KOOPMAN TENSOR

#%%
# import numpy as np
# import sys
# sys.path.append('../../')
import estimate_L
# import domain
# import observables

#%% Define domain
# bounds = np.array([[-2, 2], [-1.5, 1.5]])
# boxes = np.array([20, 15])
# Omega = domain.discretization(bounds, boxes)

#%% Define system
# def b(x):
#     return np.vstack((-4*x[0, :]**3 + 4*x[0, :], -2*x[1, :]))
 
# def sigma(x):
#     n = x.shape[1]
#     y = np.zeros((2, 2, n))
#     y[0, 0, :] = 0.7
#     y[0, 1, :] = x[0, :]
#     y[1, 1, :] = 0.5
#     return y

#%% Define observables
# order = 1
# phi = observables.monomials(order)
# psi = observables.monomials(order)

#%% Generate data
# X = Omega.randPerBox(100)
# Y = b(X)
# Z = sigma(X)

#%% Build Phi and Psi matrices
Phi_X = phi(X)
d_phi = Phi_X[:,0].shape[0]
# d_psi = phi(U[:,0]).shape[0]
d_psi = 1
N = X.shape[1]

# Phi_X = np.empty((d_phi,N))
# for i,x in enumerate(X.T):
#     Phi_X[:,i] = phi(int(x[0]))[:,0]

# Phi_Y = np.empty((d_phi,N))
# for i,y in enumerate(Y.T):
#     Phi_Y[:,i] = phi(int(y[0]))[:,0]

# dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
# if not (Z is None): # stochastic dynamical system
#     n = Phi_X.shape[0] # number of basis functions
#     ddPhi_X = phi.ddiff(X) # second-order derivatives
#     S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
#     for i in range(n):
#         dPhi_Y[i, :] += 0.5*np.sum( ddPhi_X[i, :, :, :] * S, axis=(0,1) )

# Psi_U = np.empty((d_psi,N))
# for i,u in enumerate(U.T):
#     Psi_U[:,i] = psi(int(u[0]))[:,0]

Psi_U = np.ones((d_psi,N))

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T

#%% Reshape M into K tensor
_K = np.empty((d_phi, d_phi, d_psi))
for i in range(d_phi):
    _K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, [u])


#%% Error (
#    Mean norm on training data from gedmd: 2126526417571.602
#    Mean norm on training data from koopman tensor: 9314.313990558026
# )
print("PRINT K:", K.shape)
print("PRINT M:", M.shape)

print(l2_norm(K, M))
print(l2_norm(K, M.T))

print(K[:,1])
print(M.T[:,1])

# norms = []
# norms_2 = []
# for i in range(N):
#     true_phi_x_prime = dPhi_Y[:,i]
#     K_predicted_phi_x_prime = K @ Phi_X[:,i]
#     M_predicted_phi_x_prime = K_u(_K, 1) @ Phi_X[:,i]
#     norms.append(l2_norm(true_phi_x_prime, K_predicted_phi_x_prime))
#     norms_2.append(l2_norm(true_phi_x_prime, M_predicted_phi_x_prime))
# norms = np.array(norms)
# norms_2 = np.array(norms_2)

# print("Mean norm on training data from gedmd:", norms.mean())
# print("Mean norm on training data from koopman tensor:", norms_2.mean())