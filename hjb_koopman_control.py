#%%
import numpy as np
import scipy as sp
import numba as nb
from scipy import integrate
from estimate_L import rrr

@nb.njit(fastmath=True)
def ln(x):
    return np.log(x)

#%% Dictionary functions
dictionary_functions = [lambda x: x[0], lambda x: x[1], lambda x: x[0]**2]
def psi(X):
    output = []
    for x in X:
        row = []
        for dictionary in dictionary_functions:
            row.append(dictionary(x))
        output.append(row)
    return np.array(output).T

psi([[1,2]])

#%% Variable definitions
mu = -0.1
lamb = 1
L = np.array([
    [mu, 0, 0],
    [0, lamb, -lamb],
    [0, 0, 2*mu]
])
Q = np.identity(2)
R = 1
x = np.array([
    [-5],
    [5]
])
y = np.append(x, [[25]], axis=0)
vf = lambda tau, x: L @ x

#%% Create dataset(s)
X = np.array([
    [mu * x[0,0]],
    [lamb * (x[1,0] - x[0,0]**2)]
])
for _ in range(1000):
    x = np.array([
        [mu * x[0,0]],
        [lamb * (x[1,0] - x[0,0]**2)]
    ])
    X = np.append(X, x, axis=1)
X = X[:,1:]

# Y = y.copy()
# for _ in range(1,1000):
#     y = L @ y
#     Y = np.append(Y, y, axis=1)

#%% ODE
Y = integrate.solve_ivp(vf, (0,50), y[:,0], first_step=0.05, max_step=0.05)
Y = Y.y[:,:-2]
Y_dot = vf(0, Y)

#%%
Psi_X = Y
dPsi_X = Y_dot

#%% Define a reward function from cost function
def reward(x, u):
    return -(x.T @ Q @ x + u.T * R * u)

#%% Modified learning algorithm
def learningAlgorithm(L, X, psi, Psi_X, action_bounds, reward, timesteps=100, cutoff=8, lamb=10):
    _divmax = 30
    Psi_X_T = Psi_X.T

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
        G_X = currentV.copy()
        B_v = rrr(Psi_X_T, G_X.reshape(-1,1))

        # @nb.jit(forceobj=True, fastmath=True)
        def Lv_hat(x_tilde):
            return (L @ B_v).T @ psi(x.reshape(1,-1))

        # @nb.jit(forceobj=True, fastmath=True)
        def compute(u, x):
            # print("compute")

            inp = (constant * (reward(x, u) + Lv_hat(x))).astype('longdouble')
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
            return (reward(x, u) - (lamb * ln(eval_pi_hat_star)) + Lv_hat(x)) * eval_pi_hat_star

        def V(x):
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

#%% Learn!
big = 1000
action_bounds = np.array([-big, big])
_, pi = learningAlgorithm(
    L, X, psi, Psi_X, action_bounds, reward, timesteps=4
)
# %%
