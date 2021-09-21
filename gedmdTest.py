#%% KLUS'S IMPLEMENTATAION

import numpy as np
import scipy as sp
import scipy.sparse.linalg
import domain
import observables
import matplotlib.pyplot as plt

def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

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

def printMatrix(x, name = None, k = 8):
    '''Prints the matrix like Matlab.'''
    m, n = x.shape
    c = 0
    if name != None: print(name + ' = ')
    while c < n:
        print('\033[94m  (columns %s through %s)\033[0m' % (c, min(c+k, n)-1))
        for i in range(m):
            for j in range(c, min(c+k, n)):
                print('  % 10.5f' % x[i, j], end = '')
            print('')
        c += k

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

def gedmd(X, Y, Z, phi, evs=5, operator='K'):
    '''
    Generator EDMD for the Koopman operator. The matrices X and Y
    contain the input data. For stochastic systems, Z contains the
    diffusion term evaluated in all data points X. If the system is
    deterministic, set Z = None.
    '''
    PhiX = phi(X)
    dPhiY = np.einsum('ijk,jk->ik', phi.diff(X), Y)
    if not (Z is None): # stochastic dynamical system
        n = PhiX.shape[0] # number of basis functions
        ddPhiX = phi.ddiff(X) # second-order derivatives
        S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
        for i in range(n):
            dPhiY[i, :] += 0.5*np.sum( ddPhiX[i, :, :, :] * S, axis=(0,1) )
    
    C_0 = PhiX @ PhiX.T
    C_1 = PhiX @ dPhiY.T
    if operator == 'P': C_1 = C_1.T

    A = sp.linalg.pinv(C_0) @ C_1
    
    d, V = sortEig(A, evs, which='SM')
    
    return (A, d, V)

#%% Double-well system ---------------------------------------------------------------------------

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

# generate data
X = Omega.randPerBox(100)
Y = b(X)
Z = sigma(X)

# apply generator EDMD
evs = 3 # number of eigenvalues/eigenfunctions to be computed
K, d, V = gedmd(X, Y, Z, phi, evs=evs, operator='K')
printVector(np.real(d), 'd')

# plot eigenfunctions
c = Omega.midpointGrid()
Phi_c = phi(c)
for i in range(evs):
    plt.figure(i+1)
    plt.clf()
    Omega.plot(np.real( V[:, i].T @ Phi_c ), mode='3D')

#%% system identification
# order = 4 # reduce order of monomials
# p = observables.allMonomialPowers(2, order)
# n = p.shape[1] # number of monomials up to order 4

# printMatrix(p, 'p')
# printMatrix(K[:n, :n], 'K')

# # compute entries of a evaluated in c
# b_c = K[:, 1:3].T @ Phi_c

# a_11 = K[:, 3].T @ Phi_c - 2*b_c[0, :]*c[0, :]
# a_12 = K[:, 4].T @ Phi_c - b_c[0, :]*c[1, :] - b_c[1, :]*c[0, :]
# a_22 = K[:, 5].T @ Phi_c - 2*b_c[1, :]*c[1, :]

# plt.figure(evs+1)
# Omega.plot(a_11, mode='3D')
# plt.figure(evs+2)
# Omega.plot(a_12, mode='3D')
# plt.figure(evs+3)
# Omega.plot(a_22, mode='3D')
# plt.gca().set_zlim([0, 1])

#%% Perron-Frobenius generator

# define observables
# phi = observables.gaussians(Omega, 0.2)

# # apply generator EDMD
# evs = 3 # number of eigenvalues/eigenfunctions to be computed
# P, d, V = gedmd(X, Y, Z, phi, evs=evs, operator='P')
# printVector(np.real(d), 'd')

# # plot eigenfunctions
# c = Omega.midpointGrid()
# Psi_c = phi(c)
# for i in range(evs):
#     plt.figure(i+1)
#     plt.clf()
#     Omega.plot(np.real( V[:, i].T @ Psi_c ), mode='3D')





#%% KOOPMAN TENSOR

#%%
import estimate_L

order = 10
phi = observables.monomials(order)
psi = lambda u: [1]

#%% Build Phi and Psi matrices
Phi_X = phi(X)
dim_phi = Phi_X[:,0].shape[0]
# d_psi = phi(U[:,0]).shape[0]
dim_psi = 1
N = X.shape[1]

print(Phi_X.shape)

Psi_U = np.ones((dim_psi,N))

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
dPhi_Y = np.einsum('ijk,jk->ik', phi.diff(X), Y)
n = Phi_X.shape[0] # number of basis functions
ddPhiX = phi.ddiff(X) # second-order derivatives
S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
for i in range(n):
    dPhi_Y[i, :] += 0.5*np.sum( ddPhiX[i, :, :, :] * S, axis=(0,1) )
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T

d, V = sortEig(M.T, evs, which='SM')

Phi_c = phi(c)
for i in range(evs):
    plt.figure(i+1+evs)
    plt.clf()
    Omega.plot(np.real( V[:, i].T @ Phi_c ), mode='3D')

#%% Reshape M into K tensor
# _K = np.empty((dim_phi, dim_phi, dim_psi))
# for i in range(dim_phi):
#     _K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

# def K_u(K, u):
#     return np.einsum('ijz,z->ij', K, psi(u))

# #%% Error (
# #    Mean norm on training data from gedmd: 2126526417571.602
# #    Mean norm on training data from koopman tensor: 9314.313990558026
# # )
# print("PRINT K:", K.shape)
# print("PRINT M:", M.shape)

# print(l2_norm(K, M))
# print(l2_norm(K, M.T))

# print(K[:,1])
# print(M.T[:,1])

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

plt.show()