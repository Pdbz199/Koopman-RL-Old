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
        d, V = sp.sparse.linalg.eigs(A, evs, v0=np.ones(n), which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])

""" FINITE DINFFERENCING UTILITIES """

def gradient(f, x, delta = 1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method
    Returns:
        ret (numpy.array): gradient of f at the point x
    """

    n = x.shape[0]
    ret = np.zeros(n)
    d = np.eye(n) * delta
    for i in range(n):
        ret[i] = (f(x+d[i]) - f(x-d[i])) / (2*delta)
    return ret

def jacobian(f, x, delta = 1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method
    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    
    n = x.shape[0]
    m = f(x).shape[0]
    ret = np.zeros((m, n))
    for i in range(m):
        ret[i,:] = gradient(lambda v: f(v)[i], x, delta)
    return ret

#%%