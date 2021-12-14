#%%
import numpy as np
import scipy as sp
import scipy.sparse.linalg

""" NORM UTILITIES """

def l2_norm(true_state, predicted_state):
    if true_state.shape != predicted_state.shape:
        raise Exception(f'The dimensions of the parameters did not match ({true_state.shape} and {predicted_state.shape}) and therefore cannot be compared.')
    
    err = true_state - predicted_state
    return np.sum(np.power(err, 2))

""" KOOPMAN TENSOR UTILITIES """

def K_u(K, psi_u):
    return np.einsum('ijz,zk->ij', K, psi_u)

""" KLUS'S UTILITIES """

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
# %%
