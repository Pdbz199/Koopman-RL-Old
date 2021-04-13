#%%
import observables
import numpy as np
import scipy as sp
import numba as nb
from scipy import integrate
from estimate_L import *
from cartpole_reward import cartpoleReward
@nb.njit(fastmath=True)
def ln(x):
    return np.log(x)
@nb.njit(fastmath=True) #parallel=True,
def nb_einsum(A, B):
    assert A.shape == B.shape
    res = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res += A[i,j]*B[i,j]
    return res

#%%
X = (np.load('random-cartpole-states.npy'))[:5000].T # states
U = (np.load('random-cartpole-actions.npy'))[:5000].T # actions
X_tilde = np.append(X, [U], axis=0) # extended states
d = X_tilde.shape[0]
m = X_tilde.shape[1]
s = int(d*(d+1)/2) # number of second order poly terms

#%%
psi = observables.monomials(2)
Psi_X_tilde = psi(X_tilde)
Psi_X_tilde_T = Psi_X_tilde.T
k = Psi_X_tilde.shape[0]
nablaPsi = psi.diff(X_tilde)
nabla2Psi = psi.ddiff(X_tilde)

#%%
@nb.njit(fastmath=True)
def dpsi(X, k, l, t=1):
    difference = X[:, l+1] - X[:, l]
    term_1 = (1/t) * (difference)
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * np.outer(difference, difference)
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1, term_2) + nb_einsum(term_3, term_4)

#%% Construct \text{d}\Psi_X matrix
dPsi_X_tilde = np.zeros((k, m))
for row in range(k):
    for column in range(m-1):
        dPsi_X_tilde[row, column] = dpsi(X_tilde, row, column)
dPsi_X_tilde_T = dPsi_X_tilde.T

#%%
L = rrr(Psi_X_tilde_T, dPsi_X_tilde_T)

#%%
@nb.njit
def psi_x_tilde_with_diff_u(l, u):
    result = Psi_X_tilde[:,l].copy()
    result[-1] = u
    return result

#%% Algorithm 1
#? arg for (epsilon=0.1,)?
def learningAlgorithm(L, X, Psi_X_tilde, U, reward, timesteps=100, cutoff=8, lamb=0.05):
    # placeholder functions
    V = lambda x: x
    pi_hat_star = lambda x: x

    low = np.min(U)
    high = np.max(U)

    constant = 1/lamb

    eigenvalues, eigenvectors = sp.linalg.eig(L) # L created with X_tilde
    eigenvectors = eigenvectors
    @nb.njit(fastmath=True)
    def eigenfunctions(i, psi_x_tilde):
        return np.dot(np.real(eigenvectors[i]), psi_x_tilde) #Psi_X_tilde[:, l]

    eigenvectors_inverse_transpose = sp.linalg.inv(eigenvectors).T # pseudoinverse?

    currentV = np.zeros(X.shape[1]) # V^{\pi*_0}
    lastV = currentV.copy()
    # G_X_tilde = np.empty((currentV.shape[0], currentV.shape[0]))

    # (abs(V - lastV) > epsilon).any() # there may be a more efficient way with maintaining max
    t = 0
    while t < timesteps:
        G_X_tilde = currentV.copy()
        B_v = ols(Psi_X_tilde_T, G_X_tilde)

        generatorModes = B_v.T @ eigenvectors_inverse_transpose

        @nb.jit(forceobj=True, fastmath=True)
        def Lv_hat(l, u):
            psi_x_tilde = psi_x_tilde_with_diff_u(l, u)
            summation = 0
            for ell in range(cutoff):
                summation += eigenvalues[ell] * eigenfunctions(ell, psi_x_tilde) * generatorModes[ell]
            return summation

        @nb.jit(forceobj=True, fastmath=True)
        def compute(u, l):
            inp = (constant * (reward(X[:,l], u) + Lv_hat(l, u))).astype('longdouble')
            return np.exp(inp)

        def pi_hat_star(u, l): # action given state
            numerator = compute(u, l)
            denominator = integrate.romberg(compute, low, high, args=(l,), divmax=30)
            return numerator / denominator

        def compute_2(u, l):
            eval_pi_hat_star = pi_hat_star(u, l)
            return (reward(X[:,l], u) - (lamb * ln(eval_pi_hat_star))) * eval_pi_hat_star

        def integral_summation(l):
            summation = 0
            for ell in range(cutoff):
                summation += generatorModes[ell] * eigenvalues[ell] * \
                    integrate.romberg(
                        lambda u, l: eigenfunctions(ell, Psi_X_tilde[:, l]) * pi_hat_star(u, l),
                        low, high, args=(l,), divmax=30
                    )
            return summation

        def V(l):
            return (integrate.romberg(compute_2, low, high, args=(l,), divmax=30) + \
                        integral_summation(l))

        lastV = currentV
        for i in range(currentV.shape[0]):
            currentV[i] = V(i)

        t+=1
        print("Completed learning step", t, "\n")
    
    return currentV, pi_hat_star

#%%
# V, pi = learningAlgorithm(L, X, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=3, lamb=1)
# print(pi(0, 100))
# print(pi(1, 100))

# #%% Should be 1, it is!
# print(integrate.romberg(
#     pi, 0, 1, args=(0,), divmax=30
# ))

#%% Algorithm 2
def rgEDMD(
    x_tilde,
    X_tilde,
    Psi_X_tilde,
    dPsi_X_tilde,
    k,
    z_m=np.zeros((k,k)),
    phi_m_inverse=np.linalg.inv(np.identity(k))
):
    X_tilde = np.append(X_tilde, x_tilde.reshape(-1,1), axis=1)
    Psi_X_tilde = psi(X_tilde)
    # dPsi_x_tilde = np.empty(k)
    # for k in range(k):
    #     dPsi_x_tilde[k] = dpsi(X_tilde, k, -1)
    # dPsi_X_tilde = np.append(dPsi_X_tilde, dPsi_x_tilde.reshape(-1,1), axis=1)
    for l in range(k):
        dPsi_X_tilde[l, -1] = dpsi(X_tilde, l, -2)
    dPsi_X_tilde = np.append(dPsi_X_tilde, np.zeros((k,1)), axis=1) #? should this really append 0s?

    Psi_X_tilde_m = Psi_X_tilde[:,-1].reshape(-1,1)
    Psi_X_tilde_m_T = Psi_X_tilde_m.T #? maybe pinv?

    # update z_m
    z_m = z_m + dPsi_X_tilde[:,-2].reshape(-1,1) @ Psi_X_tilde_m_T

    # update \phi_m^{-1}
    phi_m_inverse = phi_m_inverse - \
                    ((phi_m_inverse @ Psi_X_tilde_m @ Psi_X_tilde_m_T @ phi_m_inverse) / \
                        (1 + Psi_X_tilde_m_T @ phi_m_inverse @ Psi_X_tilde_m))
    
    L_m = z_m @ phi_m_inverse

    # updated dPsi_X_tilde, updated z_m, updated \phi_m^{-1}, and approximate generator
    return dPsi_X_tilde, z_m, phi_m_inverse, L_m

#%%
# dPsi_X_tilde, z_m, phi_m_inverse, L_m = rgEDMD(X_tilde[:,-1], X_tilde[:,:-1], Psi_X_tilde[:,:-1], dPsi_X_tilde[:,:-1], k)

#%% Algorithm 3
def onlineKoopmanLearning():
    global X_tilde, Psi_X_tilde, dPsi_X_tilde

    X_tilde_builder = X_tilde[:,:2]
    Psi_X_tilde_builder = Psi_X_tilde[:,:2]
    dPsi_X_tilde_builder = dPsi_X_tilde[:,:2]
    k = dPsi_X_tilde_builder.shape[0]

    z_m = np.zeros((k,k))
    phi_m_inverse = np.linalg.inv(np.identity(k))
    for x_tilde in X_tilde.T:
        dPsi_X_tilde, z_m, phi_m_inverse, L_m = rgEDMD(
            x_tilde, X_tilde_builder, Psi_X_tilde_builder, dPsi_X_tilde_builder, k, z_m, phi_m_inverse
        )

    _, pi = learningAlgorithm(L, X, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=2, lamb=1)
    return pi
    
#%% YAY!
pi = onlineKoopmanLearning()
print(pi(0, 100))
print(pi(1, 100))

#%%
