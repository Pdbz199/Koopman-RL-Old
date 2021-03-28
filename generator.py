#%%
import numpy as np
import scipy as sp

'''======================= HELPER FUNCTIONS ======================='''

# Construct B matrix as seen in 3.1.2 of the reference paper
def constructB(d, k):
    Bt = np.zeros((d, k))
    if k == 1:
        Bt[0,0] = 1
    else:
        num = np.arange(d)
        Bt[num, num+1] = 1
    B = Bt.T
    return B

# Construct similar B matrix as above, but for second order monomials
def constructSecondOrderB(s, k):
    Bt = np.zeros((s, k))
    if k == 1:
        Bt[0,0] = 1
    else:
        row = 0
        for i in range(d+1, d+1+s):
            Bt[row,i] = 1
            row += 1
    B = Bt.T
    return B

'''======================= CONSTRUCT DATA MATRIX ======================='''

from brownian import brownian

# The Wiener process parameter.
sigma = 1
# Total time.
T = 10000
# Number of steps.
N = 10000
# Time step size
dt = T/N
# Number of realizations to generate.
n = 20
# Create an empty array to store the realizations.
X = np.empty((n, N+1))
# Initial values of x.
X[:, 0] = 50
brownian(X[:, 0], N, dt, sigma, out=X[:, 1:])
Z = np.roll(X,-1)[:, :-1]
X = X[:, :-1]

#%%
'''======================= SETUP/DEFINITIONS ======================='''
import observables
from sympy import symbols
from sympy.polys.monomials import itermonomials, monomial_count
from sympy.polys.orderings import monomial_key

d = X.shape[0]
m = X.shape[1]
s = int(d*(d+1)/2) # number of second order poly terms
rtoler=1e-02
atoler=1e-02
psi = observables.monomials(2)

#%%
x_str = ""
for i in range(d):
    x_str += 'x_' + str(i) + ', '
x_syms = symbols(x_str)
M = itermonomials(x_syms, 2)
sortedM = sorted(M, key=monomial_key('grlex', np.flip(x_syms)))
# print(sortedM)

#%%
Psi_X = psi(X)
Psi_X_T = Psi_X.T
nablaPsi = psi.diff(X)
nabla2Psi = psi.ddiff(X)
print("nablaPsi Shape", nablaPsi.shape)
k = Psi_X.shape[0]

#%%
'''======================= COMPUTATIONS ======================='''

# This computes dpsi_k(x) exactly as in the paper
# t = 1 is a placeholder time step, not really sure what it should be
def dpsi(k, l, t=1):
    difference = (X[:, l+1] - X[:, l])
    term_1 = (1/t) * (difference)
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * (difference.reshape(-1, 1) @ difference.reshape(1, -1))
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1, term_2) + np.tensordot(term_3, term_4)
vectorized_dpsi = np.vectorize(dpsi)

# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(range(k), column)

#%%
# Calculate Koopman generator approximation
train = int(m * 0.8)
test = m - train
M = dPsi_X[:, :train] @ np.linalg.pinv(Psi_X[:, :train]) # \widehat{L}^\top
L = M.T # estimate of Koopman generator

def check_symmetric(a, rtol=1e-02, atol=1e-02):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
print("Is L approximately symmetric:", check_symmetric(L))

# Eigen decomposition
eig_vals, eig_vecs = sp.sparse.linalg.eigs(L) if sp.sparse.issparse(L) else sp.linalg.eig(L)
# Compute eigenfunction matrix
eig_funcs = (eig_vecs).T @ Psi_X

#%% Construct estimates of drift vector (b) and diffusion matrix (a) using two methods:
# 1. Directly from dictionary functions without dimension reduction  
# 2. Construct eigendecomposition and restrict its order

# Construct B matrix that selects first-order monomials (except 1) when multiplied by list of dictionary functions
B = constructB(d, k)
# Construct second order B matrix (selects second-order monomials)
second_orderB = constructSecondOrderB(s, k)

# Computed b function (sometimes denoted by \mu) without dimension reduction
L_times_B_transposed = (L @ B).T
def b(l):
    return L_times_B_transposed @ Psi_X[:, l] # (k,)

# Calculate Koopman modes
V_v1 = B.T @ np.linalg.inv((eig_vecs).T)

# The b_v2 function allows for heavy dimension reduction
# default is reducing by 90% (taking the first k/10 eigen-parts)
# TODO: Figure out correct place to take reals
def b_v2(l, num_dims=k//10, V=V_v1):
    res = 0
    for ell in range(k-1, k-num_dims, -1):
        res += eig_vals[ell] * eig_funcs[ell, l] * V[:, ell] #.reshape(-1, 1)
    return np.real(res)

#%%
# total_b = 0
# total_b_v2 = 0
# for l in range(m):
#     b_l = b(l)
#     b_v2_l = b_v2(l)
#     checker = np.zeros(b_l.shape)
#     total_b += np.allclose(b_l, checker, rtol=rtoler, atol=atoler)
#     total_b_v2 += np.allclose(b_v2_l, checker, rtol=rtoler, atol=atoler)
# print(total_b)
# print(total_b_v2)

#%%
# the following a functions compute the diffusion matrices as flattened vectors
# this was calculated in a weird way, so it could have issues...
L_times_second_orderB_transpose = (L @ second_orderB).T

def a(l):
    return (L_times_second_orderB_transpose @ Psi_X[:, l]) - \
        (B.T @ nablaPsi[:, :, l] @ b(l))

V_v2 = second_orderB.T @ np.linalg.inv((eig_vecs).T)
def a_v2(l):
    return (b_v2(l, V=V_v2)) - \
        (B.T @ nablaPsi[:, :, l] @ b_v2(l))

#%% Reshape a vector as matrix and perform some tests
def covarianceMatrix(a_func, l):
    a_l = a_func(l)
    covariance = np.zeros((d, d))
    row = 0
    col = 0
    covariance[row, col] = a_l[0]
    col += 1
    n = 1
    while col < d:
        covariance[row, col] = a_l[n]
        covariance[col, row] = a_l[n]
        if row == col: 
            col += 1
            row = 0
        else:
            row += 1
        n +=1
    return covariance

test = covarianceMatrix(a, 2)
test_v2 = covarianceMatrix(a_v2, 2)
print(np.diagonal(test))
print(np.diagonal(test_v2))
# print(test.shape)
# print(check_symmetric(test, 0, 0))

# for j in range(m):
#     result = np.count_nonzero(covarianceMatrix(j))
#     if result < d*d: print(result)

# diagAMat = np.zeros((d, d))
# for j in range(m):
#     evaldA = covarianceMatrix(j)
#     for i in range(d):
#         diagAMat[i, i] = evaldA[i, i]
#     result = np.allclose(diagAMat, evaldA, rtol=rtoler, atol=atoler)
#     if result: print("All close at j =", j)

# Oh no... it's not positive definite
# Some calculation must be wrong
# decomp = sp.linalg.cholesky(covarianceMatrix(0))

#%%
# A = Psi_X
# B is Joe.
B = np.zeros((d, m))
m_range = np.arange(m)
B = X[:, m_range] - Z[:, m_range]
print("B shape:", B.shape)
print("Psi_X transpose shape:", Psi_X_T.shape)
PsiMult = sp.linalg.inv(Psi_X @ Psi_X_T) @ Psi_X
C = PsiMult @ B.T
# Each col of matric C represents the coeficients in a linear combo of the dictionary functions that makes up each component of the drift vector. So each c_{} 
print("C shape:", C.shape)

b_v3 = C.T @ Psi_X
for l in range(5):
    print(b_v3[:, l])
    
def a_v3(l):
    diffusionDictCoefs = np.empty((d, d, k))
    diffusionMat = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            Bij = B[i]*B[j]
            diffusionDictCoefs[i, j] = PsiMult @ Bij
            diffusionMat[i, j] = np.dot(diffusionDictCoefs[i, j], Psi_X[:,l])
    return diffusionMat
    
# not very good ):
for l in range(5):
    print(np.diagonal(a_v3(l)))