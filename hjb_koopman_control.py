#%%
import observables
import numpy as np
import scipy as sp
import quadpy as qp
import numba as nb
import mpmath as mp
from scipy import integrate
from scipy import linalg
from estimate_L import rrr
from control import lqr

@nb.njit(fastmath=True)
def ln(x):
    return np.log(x)

def mpexp(X):
    if np.isscalar(X): return float(mp.exp(X))
    output = []
    for x in X:
        output.append(float(mp.exp(x)))
    return output

#%% Dictionary functions
psi = observables.monomials(6)

#%% Variable definitions
mu = -0.1
lamb = -0.5
A = np.array([
    [mu, 0],
    [0, lamb]
])
K = np.array([
    [mu, 0, 0],
    [0, lamb, -lamb],
    [0, 0, 2*mu]
])
B = np.array([
    [0],
    [1]
])
D_y = np.append(B, [[0]], axis=0)
Q = np.identity(2)
Q2 = np.identity(3)
R = 1

#%%
x = np.array([
    [-5],
    [5]
])
y = np.append(x, [x[0]**2], axis=0)

#%%
# C = [0.0, 0.61803399, 0.23445298] when lamb = -0.5
C = (lqr(K, D_y, Q2, R)[0])[0]
# C = np.array([0.0, 2.4142, -1.4956])
u = ((-C[:2] @ x) - (C[2] * x[0]**2))[0]

# F @ y = x
F = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

#%% Generate sample data
vf = lambda tau, x: ((A @ x.reshape(-1,1)) + np.array([[0], [-lamb * x[0]**2]]) + B*u)[:,0]
X = integrate.solve_ivp(vf, (0,50), x[:,0], first_step=0.05, max_step=0.05)
X = X.y[:,:-2]

Y = np.apply_along_axis(lambda x: np.append(x, [x[0]**2]), axis=0, arr=X)

#%%
Psi_X = psi(X)
# nablaPsi = psi.diff(X)

#%% \hat{B}
# V_X = np.zeros((1, X.shape[1]))
# B = rrr(Psi_X.T, V_X.T)

#%% Define a reward function from cost function
def reward(x, u):
    return -(x @ Q @ x + u * R * u)

#%% Modified learning algorithm
def learningAlgorithm(X, psi, Psi_X, action_bounds, reward, timesteps=4, cutoff=8, lamb=10):
    # _divmax = 20
    Psi_X_T = Psi_X.T

    # placeholder functions
    V = lambda x: x
    pi_hat_star = lambda x: x

    # constants
    n = Psi_X.shape[0]
    d = X.shape[0]
    low, high = action_bounds
    constant = 1/lamb

    # V^{\pi*_0}
    currentV = np.zeros((1, X.shape[1]))
    lastV = currentV.copy()

    t = 0
    while t < timesteps:
        V_X = currentV.copy()
        B = rrr(Psi_X_T, V_X.T)

        @nb.jit(forceobj=True, fastmath=True)
        def Lv_hat(x, u):
            nablaPsi_x = psi.diff(x.reshape(-1,1)).reshape((n, d))
            y = np.append(x, x[0]**2).reshape(-1,1)
            dy_dt = K @ y + D_y * u
            return ((nablaPsi_x.T @ B).T @ F @ dy_dt)[0,0]

        @nb.jit(forceobj=True, fastmath=True)
        def compute(u, x):
            inner = constant * (reward(x, u) + Lv_hat(x, u))
            return mpexp(inner)

        def pi_hat_star(u, x): # action given state
            numerator = compute(u, x)
            denominator = qp.quad(compute, low, high, args=(x,))[0]
            return numerator / denominator

        def compute_2(u, x):
            eval_pi_hat_star = pi_hat_star(u, x)
            return (reward(x, u) - (lamb * ln(eval_pi_hat_star)) + Lv_hat(x, u)) * eval_pi_hat_star

        def V(x):
            return qp.quad(compute_2, low, high, args=(x,))[0]

        lastV = currentV
        for i in range(currentV.shape[1]):
            x = X[:,i]
            currentV[:,i] = V(x)
            if (i+1) % 250 == 0:
                print(i+1)

        t+=1
        print("Completed learning step", t)
    
    return currentV, pi_hat_star

#%% Learn!
bound = 15
action_bounds = np.array([-bound, bound])
_, pi = learningAlgorithm(
    X, psi, Psi_X, action_bounds, reward, timesteps=5, lamb=100
)

#%%
start_state = X[:,0]
# possible_actions = [2.77, 1, 5, 12, 8.95]
possible_actions = np.arange(0, bound+0.5, 0.5)
for possible_action in possible_actions:
    print(f"{possible_action}:", pi(possible_action, start_state))
    print(f"{-possible_action}:", pi(-possible_action, start_state))
# I think -8.95 is the 'right' one

# %%
int_pi = qp.quad(pi, action_bounds[0], action_bounds[1], args=(start_state,))[0]
assert(int_pi == 1.0)






#%% Dictionary functions
psi = observables.monomials(6)

#%% Variable definitions
mu = -0.1
lamb = 1
A = np.array([
    [mu, 0],
    [0, lamb]
])
A2 = np.array([
    [mu, 0, 0],
    [0, lamb, -lamb],
    [0, 0, 2*mu]
])
B = np.array([
    [0],
    [1]
])
D_y = np.append(B, [[0]], axis=0)
Q = np.identity(2)
Q2 = np.identity(3)
R = 1

#%%
x = np.array([
    [-5],
    [5]
])
y = np.append(x, [x[0]**2], axis=0)

#%%
# C = [0.0, 0.61803399, 0.23445298] when lamb = -0.5
C = lqr(A2, D_y, Q2, R)[0][0]
# C = np.array([0.0, 2.4142, -1.4956])
u = ((-C[:2] @ x) - (C[2] * x[0]**2))[0]

# F @ y = x
F = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

#%% Generate sample data
vf = lambda tau, x: ((A @ x.reshape(-1,1)) + np.array([[0], [-lamb * x[0]**2]]) + B*u)[:,0]
X = integrate.solve_ivp(vf, (0,50), x[:,0], first_step=0.05, max_step=0.05)
X = X.y[:,:-2]

Y = np.apply_along_axis(lambda x: np.append(x, [x[0]**2]), axis=0, arr=X)

#%%
Psi_X = psi(X)

# %%
# sp.linalg.expm()