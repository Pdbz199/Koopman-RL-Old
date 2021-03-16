import numpy as np
import scipy as sp
import scipy.sparse.linalg as sparse_linalg
from scipy.spatial import distance

""" KERNEL DEFINITIONS """
class gaussianKernel(object):
    '''Gaussian kernel with bandwidth sigma.'''
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x, y):
        return np.exp(-np.linalg.norm(x-y)**2 / (2*self.sigma**2))
    def diff(self, x, y):
        return -1/self.sigma**2 * (x-y) * self(x, y)
    def ddiff(self, x, y):
        d = 1 if x.ndim == 0 else x.shape[0]
        return (1/self.sigma**4 * np.outer(x-y, x-y) - 1 / self.sigma**2 * np.eye(d)) * self(x, y)
    def laplace(self, x, y):
        return (1/self.sigma**4 * np.linalg.norm(x-y)**2 - len(x) / self.sigma**2) * self(x, y)
    def __repr__(self):
        return 'Gaussian kernel with bandwidth sigma = %f.' % self.sigma

def gramian(X, k):
    '''Compute Gram matrix for training data X with kernel k.'''
    name = k.__class__.__name__
    if name == 'gaussianKernel':
        return np.exp(-distance.squareform(distance.pdist(X.T, 'sqeuclidean'))/(2*k.sigma**2))
    elif name == 'laplacianKernel':
        return np.exp(-distance.squareform(distance.pdist(X.T, 'euclidean'))/k.sigma)
    elif name == 'polynomialKernel':
        return (k.c + X.T @ X)**k.p
    elif name == 'stringKernel':
        n = len(X)
        # compute weights for normalization
        d = np.zeros(n)
        for i in range(n):
            d[i] = k.evaluate(X[i], X[i])
        # compute Gram matrix
        G = np.ones([n, n]) # diagonal automatically set to 1
        for i in range(n):
            for j in range(i):
                G[i, j] = k.evaluate(X[i], X[j]) / np.sqrt(d[i]*d[j])
                G[j, i] = G[i, j]
        return G
    else:
        #print('User-defined kernel.')
        if isinstance(X, list): # e.g., for strings
            n = len(X)
            G = np.zeros([n, n])
            for i in range(n):
                for j in range(i+1):
                    G[i, j] = k(X[i], X[j])
                    G[j, i] = G[i, j]
        else:
            n = X.shape[1]
            G = np.zeros([n, n])
            for i in range(n):
                for j in range(i+1):
                    G[i, j] = k(X[:, i], X[:, j])
                    G[j, i] = G[i, j]
        return G


def gramian2(X, Y, k):
    '''Compute Gram matrix for training data X and Y with kernel k.'''
    name = k.__class__.__name__
    if name == 'gaussianKernel':
        #print('Gaussian kernel with sigma = %f.' % k.sigma)
        return np.exp(-distance.cdist(X.T, Y.T, 'sqeuclidean')/(2*k.sigma**2))
    elif name == 'laplacianKernel':
        #print('Laplacian kernel with sigma = %f.' % k.sigma)
        return np.exp(-distance.cdist(X.T, Y.T, 'euclidean')/k.sigma)
    elif name == 'polynomialKernel':
        #print('Polynomial kernel with degree = %f and c = %f.' % (k.p, k.c))
        return (k.c + X.T@Y)**k.p
    elif name == 'stringKernel':
        m = len(X)
        n = len(Y)
        dx = np.zeros((m,))
        dy = np.zeros((n,))
        for i in range(m):
            dx[i] = k.evaluate(X[i], X[i])
        for j in range(n):
            dy[j] = k.evaluate(Y[j], Y[j])
        
        G = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                G[i, j] = k.evaluate(X[i], Y[j]) / np.sqrt(dx[i]*dy[j])
        return G
    else:
        # print('User-defined kernel.')
        if isinstance(X, list): # e.g., for strings
            m = len(X)
            n = len(Y)
            G = np.zeros([m, n])
            for i in range(m):
                for j in range(n):
                    G[i, j] = k(X[i], Y[j])
        else:
            m = X.shape[1]
            n = Y.shape[1]
            G = np.zeros([m, n])
            for i in range(m):
                for j in range(n):
                    G[i, j] = k(X[:, i], Y[:, j])
        return G
""" END KERNEL DEFINITIONS """

""" HELPER FUNCTIONS """
def sortEig(A, evs=5, which='LM'):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.
    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = sparse_linalg.eigs(A, evs, which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])
""" END HELPER FUNCTIONS """

def kedmd(X, Y, k, regularization=0, evs=5, operator='K'):
    '''
    Kernel EDMD for the Koopman or Perron-Frobenius operator. The matrices X and Y
    contain the input data.
    :param k:        kernel, see d3s.kernels
    :param regularization:  regularization parameter
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius (note that the default is K here)
    :return:         eigenvalues d and eigenfunctions V evaluated in X
    '''
    if isinstance(X, list): # e.g., for strings
        n = len(X)
    else:
        n = X.shape[1]

    G_0 = gramian(X, k)
    G_1 = gramian2(X, Y, k)
    if operator == 'K': G_1 = G_1.T

    A = sp.linalg.pinv(G_0 + regularization * np.eye(n), rcond=1e-15) @ G_1
    d, V = sortEig(A, evs)
    if operator == 'K': V = G_0 @ V
    return (d, V)