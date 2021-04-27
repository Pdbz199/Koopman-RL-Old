#%%
import observables
import numpy as np
import scipy as sp
import numba as nb
import mpmath as mp
from scipy import integrate
from estimate_L import rrr

@nb.njit(fastmath=True)
def ln(x):
    return np.log(x)

#%% Dictionary functions
order_five_monomials = [
    lambda x: 1,
    lambda x: x[0],
    lambda x: x[1],
    lambda x: x[0]**2,
    lambda x: x[0]*x[1],
    lambda x: x[1]**2,
    lambda x: x[0]**3,
    lambda x: x[0]**2*x[1],
    lambda x: x[0]*x[1]**2,
    lambda x: x[1]**3,
    lambda x: x[0]**4,
    lambda x: x[0]**3*x[1],
    lambda x: x[0]**2*x[1]**2,
    lambda x: x[0]*x[1]**3,
    lambda x: x[1]**4,
    lambda x: x[0]**5,
    lambda x: x[0]**4*x[1],
    lambda x: x[0]**3*x[1]**2,
    lambda x: x[0]**2*x[1]**3,
    lambda x: x[0]*x[1]**4,
    lambda x: x[1]**5
]

def psi(X):
    output = []
    for x in X:
        row = []
        for function in order_five_monomials:
            row.append(function(x))
        output.append(row)
    return np.array(output).T

#%% Variable definitions
mu = -0.1
lamb = 1
A = np.array([
    [mu, 0, 0],
    [0, lamb, -lamb],
    [0, 0, 2*mu]
])
B = np.array([
    [0],
    [1],
    [0],

    [0],
    [0]
])
Q = np.identity(2)
R = 1

x = np.array([
    [-5],
    [5]
])
y = psi(x.reshape(1,-1))

#%% New formulation
C = np.array([0,2.4142,-1.4956])
u = ((-C[0:2] @ x) - (C[2] * x[0]**2))[0]
y_tilde = np.append(y, [[u], [u*y[0,0]]], axis=0)

M = np.array([
    [mu, 0, 0, B[0,0], 0],
    [0, lamb, -lamb, B[1,0], 0],
    [0, 0, 2*mu, 0, 2*B[0,0]],

    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
# M is applied to extended dictionary space (y_tilde), i.e. [y1, y2, y3, u, u*y1].T
L = M.T
# L =
# [mu,     0,      0       ,   0, 0],
# [0,      lamb,   0       ,   0, 0],
# [0,      -lamb,  2*mu    ,   0, 0],
# [B[0,0], B[1,0], 0       ,   0, 0],
# [0,      0,      2*B[0,0],   0, 0]
# which is applied to the dictionary space of x (y), i.e. [y1, y2, y3].T

# E @ y_tilde = x
E = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])

#%% ODE
vf = lambda tau, x: ((M @ x.reshape(-1,1)) + B*u)[:,0]
Y_tilde = integrate.solve_ivp(vf, (0,50), y_tilde[:,0], first_step=0.05, max_step=0.05)
Y_tilde = Y_tilde.y[:,:-2]
Y_tilde_dot = vf(0, Y_tilde[:,0])

#%%
Psi_X_tilde = Y_tilde
dPsi_X_tilde = Y_tilde_dot

#%%
# B.T @ Psi = V
# || V_X - B.T @ Psi_X_tilde ||
V_X = np.zeros((1,Psi_X_tilde.shape[1]))
B = rrr(Psi_X_tilde.T, V_X.T)






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

    eigenvalues, eigenvectors = sp.linalg.eig(L)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    currentV = np.zeros(X.shape[1]) # V^{\pi*_0}
    lastV = currentV.copy()

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

            return mp.exp(constant * (reward(x, u) + Lv_hat(x)))

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
bound = 1000
action_bounds = np.array([-bound, bound])
_, pi = learningAlgorithm(
    L, X, psi, Psi_X, action_bounds, reward, timesteps=4
)
# %%
