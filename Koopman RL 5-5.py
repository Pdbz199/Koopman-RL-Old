'''======= OUTLINE ======='''
# LQR in Python
# Reformulation of LQR in framework
# VI for LQR
# Current issues with our implementation
# Ways forward
#   Model Predictive Path Integral (MPPI) locally with Koopman dynamics
#   Local LQR with
'''===== END OUTLINE ====='''

# IMPORTS
import observables
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from control import lqr


# Slow Manifold Example (KOOC/KRONIC)
## Model setup
'''
In this section, we study a deterministic dynamical system where we can explicitly find a finite dimensional linear embedding for the Koopman operator. This example comes from Steve's KRONIC paper. In particular, we consider a controlled system with quadratic nonlinearity that gives rise to a slow manifold:

\begin{align}
    \frac{d}{dt}
    \begin{bmatrix}
        x_1\\
        x_2
    \end{bmatrix} = 
    \begin{bmatrix}
        \mu x_1
        \\
        \lambda (x_2 - x_1^2)
    \end{bmatrix}
    + \mathbf{D} \ u
\end{align}
Considering a transformation to a Koopman invariant subspace, $(y_1,y_2,y_3) = (x_1,x_2,x_1^2)$, we can express the system dynamics as
\begin{align}
    \frac{d}{dt}
    \begin{bmatrix}
        y_1\\
        y_2\\
        y_3
    \end{bmatrix} = 
    \begin{bmatrix}
        \mu & 0 & 0
        \\
        0 & \lambda & -\lambda
        \\
        0 & 0 & 2\mu
    \end{bmatrix}
    \begin{bmatrix}
        y_1\\
        y_2\\
        y_3
    \end{bmatrix}
    +
    \underbrace{\begin{bmatrix}
        1 & 0\\
        0 & 1\\
        2y_1 & 0
    \end{bmatrix}
    \mathbf{D}}_{D_y} \ u
\end{align}
'''
## Definitions
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

## Calculate LQR controller
# C is supposed to be [0,2.4142] and
# C2 is supposed to be [0,2.4142,-1.4956]

C = lqr(A, B, Q, R)[0][0]
vf = lambda tau, x: A @ x + np.array([0, -lamb * x[0]**2]) + B.T * (-C @ x)
xLQR = integrate.solve_ivp(vf, (0,50), x0[:,0], first_step=0.01, max_step=0.01)
xLQR = xLQR.y[:,:-2]

C2 = lqr(A2, B2, Q2, R)[0][0]
vf2 = lambda tau, x: A @ x + np.array([0, -lamb * x[0]**2]) + B.T * ((-C2[0:2] @ x) - (C2[2] * x[0]**2))
xKOOC = integrate.solve_ivp(vf2, (0,50), x0[:,0], first_step=0.01, max_step=0.01)
xKOOC = xKOOC.y[:,:-2]

JLQR = np.cumsum(xLQR[0]**2 + xLQR[1]**2 + (C @ np.conjugate(xLQR))**2)
JKOOC = np.cumsum(xKOOC[0]**2 + xKOOC[1]**2 + (C @ np.conjugate(xKOOC))**2)


## Plots
plt.style.use('dark_background')

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

## Koopman MDP Approach
'''
Let $\mathbf{F}\ y = x$,
\begin{align}
    \mathbf{F} = \begin{bmatrix}
        1 & 0 & 0\\
        0 & 1 & 0
    \end{bmatrix}
    \implies
    \frac{\text{d}x}{\text{d}t} = \mathbf{E}\ \frac{\text{d}y}{\text{d}t}
\end{align}
Note: $\exists$ linear transformation $y \to x$ but not $x \to y$\\\\
Now looking at the continuation term of the HJB equation:
\begin{align}
    \mathcal{L} V = \dot{V} = \frac{\text{d}V}{\text{d}x}\ \frac{\text{d}x}{\text{d}t} = \nabla_x V\ \mathbf{F}\ \frac{\text{d}y}{\text{d}t}
\end{align}
In order to characterize $\dot{V}$, we must then derive an expression for $\frac{\text{d}V}{\text{d}x}$. Assuming that $V$ is a linear combination of the Koopman invariant subspace $y$:
\begin{align}
    V = \mathbf{B}^\top\ y \implies \nabla_x V &= \nabla_x \mathbf{B}^\top\ y\\
    &= \nabla_x (y^\top\ \mathbf{B})\\
    &= J_y (x_1, x_2)\ \mathbf{B}\\
    &= \begin{bmatrix}
        \frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \frac{\partial y_3}{\partial x_1}\\
        \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \frac{\partial y_3}{\partial x_2}
    \end{bmatrix}
    \ \mathbf{B}
\end{align}

If it is correct to think statically (holding time fixed), then the Jacobian matrix is not hard to calculate and is given by
\begin{align*}
    J_y (x_1, x_2) =
    \begin{bmatrix}
        \frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \frac{\partial y_3}{\partial x_1}\\
        \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \frac{\partial y_3}{\partial x_2}
    \end{bmatrix}
    &= \begin{bmatrix}
        \mu & 0 & 2x_1\\
        0 & 1 & 0
    \end{bmatrix}
\end{align*}


Because LQR has no discounting and it is framed as a minimization problem, the HJB becomes
\begin{align}
    0 &=  \inf_{\pi_t \in \mathcal{P}(U)} \underbrace{
    \int_U \left( r(x,u) + \lambda \ln{\pi_t(u)} + (\mathcal{L}v)(x,u)\right) \pi_t(u) \text{d}u}_{H\left(x,\pi(x),\mathcal{L}V\right)} \label{HJB_LQR}
\end{align}
where $r(x,u) = x^\top Q x + u^\top Ru$ and the function $H$ denotes the Hamiltonian to be minimized. The optimal policy is then
\begin{align}
    \pi(x) = \argmin_{\pi} H\left(x,\pi(x),\mathcal{L}V\right)
\end{align}
the solution of which is
\begin{align}
    \pi^*(u|x) &= \frac{\exp\left(-\frac{1}{\lambda}(r(x,u) + (\mathcal{L}v)(x,u))\right)}{\int_U \exp\left(-\frac{1}{\lambda}(r(x,u) + (\mathcal{L}v)(x,u))\right)du}\label{opt_pol_LQR}
\end{align}
We can use the DP relationship to update the value function as follows
\begin{align}
    V_{j+1}(x(t)) = \mathbb{E}_{u\sim\pi}\left(\int_{t}^{t+\tau} r(x(s),u)ds\right) + V_j(x(t+\tau))
\end{align}
Expressing this in terms of the Koopman operator
\begin{align}
    V_{j+1}(x(t)) = \mathbb{E}_{u\sim\pi}\left(\int_{t}^{t+\tau} r(x(s),u)ds\right) + \mathcal{K}_\tau V_j(x(t))
\end{align}
Using the relationship between the Koopman operator and its generator $\mathcal{K}_{\tau} = \exp(\tau\mathcal{L})$
\begin{align}
    V_{j+1}(x(t)) &= \mathbb{E}_{u\sim\pi}\left(\int_{t}^{t+\tau} r(x(s),u)ds\right) + \exp(\tau\mathcal{L}) V_j(x(t))
    \\
    &= \int_U\left(\int_{t}^{t+\tau} r(x(s),u)ds\right)\pi(u)du + \exp(\tau\mathcal{L}) V_j(x(t))
    \\
     &= \int_{t}^{t+\tau}\left( \int_U r(x(s),u)\pi(u)du\right)ds + \exp(\tau\mathcal{L}) V_j(x(t))
\end{align}

Specializing the reward to the special case of LQR $r(x,u) = x^\top Q x + u^\top Ru$ 
\begin{align}
    V_{j+1}(x(t))
     &= \int_{t}^{t+\tau}x(t)^\top Q x(t)ds + \tau\int_U  u^\top Ru \;\pi(u)du + \exp(\tau\mathcal{L}) V_j(x(t))
\end{align}
Furthermore, if we take the control to be univariate
\begin{align}
    V_{j+1}(x(t))
     &= \int_{t}^{t+\tau}x(t)^\top Q x(t)ds + \tau \mathbb{E}_{u\sim\pi}(Ru^2) + \exp(\tau\mathcal{L}) V_j(x(t)) \label{BellmanEq}
\end{align}
Next, we outline an algorithm for value function iteration in this case

\begin{enumerate}
    \item Simulate states by taking (for now) the optimal policy of $u$ from the KOOC approach as given and using the dynamics of x. This gives us samples $\bf{x_1,...,x_m}$ and $u_1,...,u_{m-1}$. In addition, we also have observations $\bf{y_1,...,y_m}$.
    
    \item Guess initial value function and evaluate it at sample points
    
    \item Use least squares to project onto dictionary space (which is richer than the minimal function space that spans the dynamics). Denote this matrix by $\hat B^\top$.
    
    \item Get the unforced Koopman operator of ${\bf y}=(x_1,x_2,x_1^2)$
    
    \item Calculate $\hat{\mathcal{L}} V = \dot{V} = \frac{\text{d}V}{\text{d}x}\ \frac{\text{d}x}{\text{d}t} = \nabla_x \psi(x)^\top \hat B\ \mathbf{F}\ \frac{\text{d}y}{\text{d}t}$
    
    \item Find opt policy $\pi^*$ using \eqref{opt_pol_LQR}
    
    \item Calculate next value function by plugging in optimal policy and the current value function into the Bellman equation \eqref{BellmanEq}
\end{enumerate}
Unfortunately, dealing with the numerical issues with integration is proving to be difficult, especially when evaluating the optimal policy on each iteration and integrating wrt it.
'''