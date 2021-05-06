#%%
import observables
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from control import lqr

#%% Definitions
mu = -0.1
lamb = 1
tspan = np.arange(0, 50, 0.01)
x0 = np.array([
    [-5],
    [5]
])

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

A2 = np.array(
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

#%% Traditional
C = lqr(A, B, Q, R)[0][0]
# C = np.array([0,2.4142])
vf = lambda tau, x: A @ x + np.array([0, -lamb * x[0]**2]) + B.T * (-C @ x)
xLQR = integrate.solve_ivp(vf, (0,50), x0[:,0], first_step=0.01, max_step=0.01)
xLQR = xLQR.y[:,:-2]

#%% Koopman
C2 = lqr(A2, B2, Q2, R)[0][0]
# C2 = np.array([0,2.4142,-1.4956])
vf2 = lambda tau, x: A @ x + np.array([0, -lamb * x[0]**2]) + B.T * ((-C2[0:2] @ x) - (C2[2] * x[0]**2))
xKOOC = integrate.solve_ivp(vf2, (0,50), x0[:,0], first_step=0.01, max_step=0.01)
xKOOC = xKOOC.y[:,:-2]

JLQR = np.cumsum(xLQR[0]**2 + xLQR[1]**2 + (C @ np.conjugate(xLQR))**2)
JKOOC = np.cumsum(xKOOC[0]**2 + xKOOC[1]**2 + (C @ np.conjugate(xKOOC))**2)

#%% Plots
plt.style.use('dark_background')

# 3 horizontal sub plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.plot(xLQR[0], xLQR[1], 'w-')
ax1.plot(xKOOC[0], xKOOC[1], 'r--')
ax1.grid(b=True, which='major', color='#666666', linestyle='-')
ax1.set_xlabel('x_1')
ax1.set_ylabel('x_2')

ax2.set_xlim(0, 50)
ax2.plot(tspan, xLQR.T, 'w-')
ax2.plot(tspan, xKOOC.T, 'r--')
ax2.grid(b=True, which='major', color='#666666', linestyle='-')
ax2.set_xlabel('t')
ax2.set_ylabel('x_k')

ax3.set_xlim(0, 50)
ax3.plot(tspan, JLQR, 'w-')
ax3.plot(tspan, JKOOC, 'r--')
ax3.grid(b=True, which='major', color='#666666', linestyle='-')
ax3.set_xlabel('t')
ax3.set_ylabel('J')

plt.subplots_adjust(wspace=0.5)
plt.show()

# %%