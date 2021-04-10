#%%
# from base import * #np, sp, n, B

#%%
import observables
import numpy as np
import scipy as sp
from scipy import integrate
from estimate_L import *
from cartpole_reward import cartpoleReward
def ln(x):
    return np.log(x)

#%%
X = (np.load('cartpole-states.npy'))[:5000].T # states
U = (np.load('cartpole-actions.npy'))[:5000].T # actions
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
def dpsi(k, l, t=1):
    difference = (X_tilde[:, l+1] - X_tilde[:, l])
    term_1 = (1/t) * (difference)
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * (difference.reshape(-1, 1) @ difference.reshape(1, -1))
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1, term_2) + np.einsum('ij,ij->', term_3, term_4)
vectorized_dpsi = np.vectorize(dpsi)

#%% Construct \text{d}\Psi_X matrix
dPsi_X_tilde = np.empty((k, m))
for column in range(m-1):
    dPsi_X_tilde[:, column] = vectorized_dpsi(np.arange(k), column)
dPsi_X_tilde_T = dPsi_X_tilde.T

#%%
L = rrr(Psi_X_tilde_T, dPsi_X_tilde_T)

#%%
# arg for (epsilon=0.1,)
def learningAlgorithm(L, X, Psi_X_tilde, U, reward, timesteps=100, cutoff=8, lamb=0.05):
    # placeholder functions
    V = lambda x: x
    pi_hat_star = lambda x: x

    low = np.min(U)
    high = np.max(U)

    constant = 1/lamb

    eigenvalues, eigenvectors = sp.linalg.eig(L) # L created with X_tilde
    eigenfunctions = lambda ell, l: np.dot(eigenvectors[ell], Psi_X_tilde[:, l])

    eigenvectors_inverse_transpose = sp.linalg.inv(eigenvectors).T # pseudoinverse?

    # j = 1 # TODO: is this j index useful?
    currentV = np.zeros(X.shape[1]) # V^{\pi*_0}
    lastV = currentV.copy()
    G_X_tilde = np.empty((currentV.shape[0], currentV.shape[0]))

    # (abs(V - lastV) > epsilon).any() # there may be a more efficient way with maintaining max
    t = 0
    while t < timesteps:
        G_X_tilde[np.arange(currentV.shape[0])] = currentV.copy()
        B_g = rrr(Psi_X_tilde.T, G_X_tilde.T)

        generatorModes = B_g.T @ eigenvectors_inverse_transpose

        def Lv_hat(l):
            summation = 0
            for ell in range(cutoff):
                summation += eigenvalues[ell] * eigenfunctions(ell, l) * generatorModes[ell]
            return summation

        compute = lambda u, l: np.exp(constant * (reward(X[:,l], u) + Lv_hat(l)))

        def pi_hat_star(u, l): # action given state
            numerator = compute(u, l)
            denominator = integrate.quad(compute, low, high, args=(l))
            return numerator / denominator
        
        def integral_summation(l):
            for ell in range(cutoff):
                generatorModes[ell] * eigenvalues[ell] * \
                    integrate.quad(
                        lambda u, l: eigenfunctions(ell, np.append(X[:,l], u, axis=0)) * pi_hat_star(u, l),
                        low, high, args=(l)
                    )

        def V(l):
            return integrate.quad(
                lambda u, l: (reward(X[:,l], u) - (lamb * ln(pi_hat_star(u, l)))) * pi_hat_star(u, l),
                low, high, args=(l)
            ) + integral_summation(l)

        lastV = currentV
        for i in range(currentV.shape[0]):
            currentV[i] = V(i)

        t+=1
        # j+=1
    
    return V, pi_hat_star

#%%
V, pi = learningAlgorithm(L, X, Psi_X_tilde, [0,1], cartpoleReward, timesteps=1)

#%% Rough attempt at Algorithm 2
n = x.shape[0]
delta = 1
phi = delta * np.identity(n)
z = np.zeros((n,n))
def rgEDMD(x):
    global Psi_X, dPsi_X, phi, z

    X = np.append(X, x, axis=1)
    Psi_X = psi(X)
    dPsi_X = np.append(dPsi_X, dpsi(k, X.shape[1]-1), axis=1)

    Psi_X_pinv = sp.linalg.pinv(Psi_X)
    z = z + dPsi_X @ Psi_X_pinv
    phi_inverse = sp.linalg.inv(phi)
    phi_inverse = phi_inverse - \
                    ((phi_inverse @ Psi_X @ Psi_X_pinv @ phi_inverse) / (1 + Psi_X_pinv @ phi_inverse @ Psi_X))
    L_m = z @ phi_inverse

    return L_m

# %%
