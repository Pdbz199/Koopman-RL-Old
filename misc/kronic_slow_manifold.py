#%% Imports
import matplotlib.pyplot as plt
import numpy as np

from control import lqr
from scipy import integrate

#%% System variable definitions
mu = -0.1
# lamb = -0.5
lamb = 1.0

A = np.array([
    [mu, 0   ],
    [0,  lamb]
])
B = np.array([
    [0],
    [1]
])
Q = np.identity(2)
R = 1

K = np.array(
    [[mu, 0,    0    ],
     [0,  lamb, -lamb],
     [0,  0,    2*mu ]]
)
B2 = np.array(
    [[0],
     [1],
     [0]]
)
Q2 = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
)

#%%
initial_x = np.array([
    [-5.],
    [5.]
])
initial_y = np.append(initial_x, [initial_x[0]**2], axis=0)

#%% Standard LQR
C = lqr(A, B, Q, R)[0][0]
print("Standard LQR:", C)
# C = np.array([0,2.4142]) when lamb = 1

#%% Koopman LQR
# C = [0.0, 0.61803399, 0.23445298] when lamb = -0.5
C2 = lqr(K, B2, Q2, R)[0][0]
print("Koopman LQR:", C2)
# C = np.array([0.0, 2.4142, -1.4956])
# u = ((-C[:2] @ x) - (C[2] * x[0]**2))[0]

def vf(tau, x, u):
    x = np.vstack(x)
    u = u.reshape(1, -1)

    return ((A @ x) + np.array([[0], [-lamb * x[0, 0]**2]]) + B @ u)[:, 0]

def C2_func(x):
    # np.random.seed( np.abs( int( hash(str(x)) / (10**10) ) ) )
    return np.array([
        np.random.uniform(-100,100),
        np.random.uniform(-100,100),
        np.random.uniform(-100,100)
    ])

def U_builder(X):
    U = []
    for x in X.T:
        U.append([-C2_func(x)@x])
    return np.array(U).T

#%% Randomly controlled system (currently takes 5-ever to run
# X = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2_func(x)[:2] @ x) - (C2_func(x)[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
# X = X.y
# Y = np.roll(X, -1, axis=1)[:,:-1]
# X = X[:,:-1]
# U = U_builder(X)

#%% Standard LQR controlled system
X_opt = integrate.solve_ivp(
    fun=lambda tau, x: vf(tau, x, (-C @ x)),
    t_span=(0, 50),
    y0=initial_x[:, 0],
    first_step=0.05,
    max_step=0.05
)
X_opt = X_opt.y # (2, 1001)
Y_opt = np.roll(X_opt, -1, axis=1)[:,:-1]
X_opt = X_opt[:,:-1]
U_opt = (-C @ X_opt).reshape(1, -1)

#%% Koopman LQR controlled system
X2_opt = integrate.solve_ivp(
    fun=lambda tau, x: vf(tau, x, ((-C2[:2] @ x) - (C2[2] * x[0]**2))),
    t_span=(0, 50),
    y0=initial_x[:, 0],
    first_step=0.05,
    max_step=0.05
)
X2_opt = X2_opt.y # (2, 1001)
Y2_opt = np.roll(X2_opt, -1, axis=1)[:,:-1]
X2_opt = X2_opt[:,:-1]
U2_opt = (-C @ X2_opt).reshape(1, -1)

#%% Plot the trajectories
fig = plt.figure()
plt.suptitle("Slow Manifold Dynamics")

ax = fig.add_subplot(1, 1, 1)
ax.set_title("Standard LQR v. KRONIC")
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.plot(X_opt[0], X_opt[1])
ax.plot(X2_opt[0], X2_opt[1])

plt.tight_layout()
plt.show()