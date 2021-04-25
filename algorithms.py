#%%
import observables
import numpy as np
import scipy as sp
import numba as nb
from scipy import integrate, interpolate
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
# Columns are state vectors
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
# @nb.njit
# def psi_x_tilde_with_diff_u(psi_X_tilde, u):
#     result = psi_X_tilde
#     result[-1] = u
#     return result

#%% Algorithm 1
#? arg for (epsilon=0.1,)?
# TODO: Make sure np.real() calls are where they need to be
def learningAlgorithm(L, X, psi, Psi_X_tilde, action_bounds, reward, timesteps=100, cutoff=8, lamb=10):
    _divmax = 30
    Psi_X_tilde_T = Psi_X_tilde.T

    # placeholder functions
    V = lambda x: x
    pi_hat_star = lambda x: x

    low = action_bounds[0]
    high = action_bounds[1]

    constant = 1/lamb

    eigenvalues, eigenvectors = sp.linalg.eig(L) # L created with X_tilde
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    # @nb.njit(fastmath=True)
    # def eigenfunctions(ell, psi_x_tilde):
    #     return np.dot(eigenvectors[ell], psi_x_tilde)[0]

    # eigenvectors_inverse_transpose = sp.linalg.inv(eigenvectors).T # pseudoinverse?

    currentV = np.zeros(X.shape[1]) # V^{\pi*_0}
    lastV = currentV.copy()
    # G_X_tilde = np.empty((currentV.shape[0], currentV.shape[0]))

    # (abs(V - lastV) > epsilon).any() # there may be a more efficient way with maintaining max
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html
    # scipy.integrate.RK45(fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None, **extraneous)
    # Not really sure how to replace romberg with that...
    t = 0
    while t < timesteps:
        G_X_tilde = currentV.copy()
        B_v = rrr(Psi_X_tilde_T, G_X_tilde.reshape(-1,1))

        # generatorModes = B_v.T @ eigenvectors_inverse_transpose

        # @nb.jit(forceobj=True, fastmath=True)
        def Lv_hat(x_tilde):
            # print("Lv_hat")

            # psi_x_tilde = psi(x_tilde.reshape(-1,1))
            # summation = 0
            # for ell in range(cutoff):
            #     summation += generatorModes[ell] * eigenvalues[ell] * eigenfunctions(ell, psi_x_tilde)
            # return summation

            return (L @ B_v).T @ psi(x_tilde.reshape(-1,1))

        # @nb.jit(forceobj=True, fastmath=True)
        def compute(u, x):
            # print("compute")

            x_tilde = np.append(x, u)
            inp = (constant * (reward(x, u) + Lv_hat(x_tilde))).astype('longdouble')
            # in discrete setting: 
            # p_i \propto exp(x_i)
            # p_i \propto exp(x_i - \sum_i x_i / d)
            return np.exp(inp)

        def pi_hat_star(u, x): # action given state
            # print("pi_hat_star")

            numerator = compute(u, x)
            denominator = integrate.romberg(compute, low, high, args=(x,), divmax=_divmax)
            return numerator / denominator

        def compute_2(u, x):
            # print("compute_2")

            eval_pi_hat_star = pi_hat_star(u, x)
            x_tilde = np.append(x, u)
            return (reward(x, u) - (lamb * ln(eval_pi_hat_star)) + Lv_hat(x_tilde)) * eval_pi_hat_star

        # def integral_summation(x):
        #     # print("integral_summation")

        #     summation = 0
        #     for ell in range(cutoff):
        #         summation += generatorModes[ell] * eigenvalues[ell] * \
        #             integrate.romberg(
        #                 lambda u, x: eigenfunctions(ell, psi(np.append(x, u).reshape(-1,1))) * pi_hat_star(u, x),
        #                 low, high, args=(x,), divmax=_divmax
        #             )
        #     return summation

        def V(x):
            # print("V")

            # return (integrate.romberg(compute_2, low, high, args=(x,), divmax=_divmax) + \
            #             integral_summation(x))

            return integrate.romberg(
                compute_2, low, high, args=(x,)
            )

        lastV = currentV
        for i in range(currentV.shape[0]):
            x = X[:,i]
            currentV[i] = V(x)
            if i % 1000 == 0:
                print(i)

        t+=1
        print("Completed learning step", t, "\n")
    
    return currentV, pi_hat_star

#%%
# V, pi = learningAlgorithm(L, X, psi, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=4, lamb=10)
# action = np.array([2.3, 1.0, 10.0, 1.0])
# print(pi(0, action))
# print(pi(1, action))

# %%
# 1. For each new state, x, evaluate the density on two regions [0,0.5) and (0.5,1]. This gives you the piecewise constant density on the two regions.
#    Call these f1 and f2
# 2. The inverse cdf is then given by F^{-1}(y) = y/f_1 for y\in[0,0.5f1] and (y-0.5*f1 +0.5*f2)/f2 for y\in (0.5*f1, 0.5*f1+0.5*f2]
# 3. Draw a uniform(0,1) random sample y and evaluate F^{-1}(y) to get the sample from the \pi(u|x)
# 4. Given sampled action, update state and repeat

# def sample_action(x):
#     f1 = pi(0.2, x)
#     f2 = pi(0.7, x)
#     y = np.random.rand()
#     if y <= 0.5*f1 and y >= 0:
#         return y / f1
    
#     return (y - 0.5 * f1 + 0.5 * f2) / f2

# #%%
# sample_action(100)

# from scipy import stats
# Didn't work:
# class your_distribution(stats.rv_continuous):
#     def _pdf(self, u):
#         return pi(u, 100)
# distribution = your_distribution()
# print(distribution.rvs())

# def rejection_sampler(p, xbounds, pmax):
#     while True:
#         x = np.random.rand(1)*(xbounds[1]-xbounds[0])+xbounds[0]
#         y = np.random.rand(1)*pmax
#         if y<=p(x[0]):
#             return x
# rejection_sampler(lambda u: pi(u, 100), [0,1], 1.1)

# sample from gaussian distribution instead?
# second order taylor expansion around mean of pi (things inside exp)
# damping in control

# %% Should be 1, it is!
# print(integrate.romberg(
#     pi, 0, 1, args=(0,), divmax=30
# ))

#%% Algorithm 2
def rgEDMD(
    dPsi_X_tilde,
    Psi_X_tilde_m,
    z_m,
    phi_m_inverse
):
    # update z_m
    z_m = z_m + dPsi_X_tilde[:,-2].reshape(-1,1) @ Psi_X_tilde_m.T

    # update \phi_m^{-1}
    phi_m_inverse = phi_m_inverse - \
                    ((phi_m_inverse @ Psi_X_tilde_m @ Psi_X_tilde_m.T @ phi_m_inverse) / \
                        (1 + Psi_X_tilde_m.T @ phi_m_inverse @ Psi_X_tilde_m))
    
    L_m = z_m @ phi_m_inverse

    # updated z_m, updated \phi_m^{-1}, and \mathcal{L}_m
    return z_m, phi_m_inverse, L_m

#%% Algorithm 3
# running this would take an infeasible amount of time to so instead,
# comment out line the learningAlgorithm call in the loop and uncommment
# the learningAlgorithm call outside of the loop for testing
@nb.jit(forceobj=True, fastmath=True)
def onlineKoopmanLearning(X_tilde, Psi_X_tilde, dPsi_X_tilde):
    X_tilde_builder = X_tilde[:,:2]
    Psi_X_tilde_builder = Psi_X_tilde[:,:2]
    dPsi_X_tilde_builder = dPsi_X_tilde[:,:2]
    k = dPsi_X_tilde_builder.shape[0]

    z_m = np.zeros((k,k))
    phi_m_inverse = np.linalg.inv(np.identity(k))
    for x_tilde in X_tilde:
        x_tilde = x_tilde.reshape(-1,1)

        X_tilde_builder = np.append(X_tilde_builder, x_tilde, axis=1)
        Psi_X_tilde_builder = np.append(Psi_X_tilde_builder, psi(x_tilde), axis=1)
        for l in range(k):
            dPsi_X_tilde_builder[l, -1] = dpsi(X_tilde_builder, l, -2)
        dPsi_X_tilde_builder = np.append(dPsi_X_tilde_builder, np.zeros((k,1)), axis=1)

        Psi_X_tilde_m = Psi_X_tilde[:,-1].reshape(-1,1)

        z_m, phi_m_inverse, L_m = rgEDMD(
            dPsi_X_tilde_builder, Psi_X_tilde_m, z_m, phi_m_inverse
        )
        # _, pi = learningAlgorithm(L, X, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=2, lamb=1)

    _, pi = learningAlgorithm(L, X, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=2, lamb=1)
    return pi