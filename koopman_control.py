#%%
import observables
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#%% Definitions
mu = -0.1
lamb = 1
tspan = np.arange(0, 50, 0.01)
x0 = np.array([
    [-5],
    [5]
])

A = np.array([
    [mu, 0],
    [0, lamb]
])
B = np.array([
    [0],
    [1]
])

#%% Traditional
C = np.array([0,2.4142])
vf = lambda tau, x: A @ x + np.array([0, -lamb * x[0]**2]) + B.T * (-C @ x)
xLQR = integrate.solve_ivp(vf, (0,50), x0[:,0], first_step=0.01, max_step=0.01)
xLQR = xLQR.y[:,:-2]

#%% Koopman
C2 = np.array([0,2.4142,-1.4956])
vf2 = lambda tau, x: A @ x + np.array([0, -lamb * x[0]**2]) + B.T * ((-C2[0:2] @ x) - (C2[2] * x[0]**2))
xKOOC = integrate.solve_ivp(vf2, (0,50), x0[:,0], first_step=0.01, max_step=0.01)
xKOOC = xKOOC.y[:,:-2]

#%% Plots
plt.style.use('dark_background')

plt.plot(xLQR[0], xLQR[1], 'w-')
plt.plot(xKOOC[0], xKOOC[1], 'r--')

plt.plot(tspan, xLQR.T, 'w-')
plt.plot(tspan, xKOOC.T, 'r--')
plt.xlim(0, 50)

JLQR = np.cumsum(xLQR[0]**2 + xLQR[1]**2 + (C @ np.conjugate(xLQR))**2)
JKOOC = np.cumsum(xKOOC[0]**2 + xKOOC[1]**2 + (C @ np.conjugate(xKOOC))**2)
plt.plot(tspan, JLQR, 'w-')
plt.plot(tspan, JKOOC, 'r--')

plt.grid(True)
plt.show()