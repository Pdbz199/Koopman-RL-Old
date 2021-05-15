import math
import observables
import numpy as np
import scipy as sp
import numba as nb
from scipy import integrate
from estimate_L import *
from cartpole_reward import cartpoleReward

#Brownian Updates



## Condition number
print("Condition number of L:", np.linalg.cond(L)) # inf
print("Condition number of Phi_X:", np.linalg.cond(Psi_X)) # 98 million+

## RRR
"""
Using the fact that OLS is essentially orthogonal projectionon the column space of X, we can rewrite L as
\begin{align}
    L = ||Y - X\hat{B}_{\text{OLS}}||^2 + ||X\hat{B}_{\text{OLS}} - XB||^2
\end{align}
The first term does not depend on B and the second term can be minimized by the SVD/PCA of the fitted values $\hat{Y} = X\hat{B}_{\text{OLS}}$
Specifically, if $U_r$ are the first r prinicpal axes of \hat{Y}, then
\begin{align}
    \hat{B}_{\text{RRR}} = \hat{B}_{\text{OLS}} U_r U_r^\top
\end{align}

MAYBE USEFUL TO INCLUDE MORE?
First, one can use it for regularization purposes. Similarly to ridge regression (RR), lasso, etc., RRR introduces some "shrinkage" penalty on B.
The optimal rank r can be found via cross-validation. In my experience, RRR easily outperforms OLS but tends to lose to RR.
However, RRR+RR can perform (slightly) better than RR alone.
"""
@nb.njit(fastmath=True)
def rrr(X, Y, rank=8):
    B_ols = ols(X, Y)
    U, S, V = np.linalg.svd(Y.T @ X @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

## CartPole Reward
"""
The default CartPole reward is always 1 until the episode terminates.
We needed to modify this to make sense in the scope of our design so we found a variable reward formulation.
We defined it below and put it into the CartPole environment for an agent to learn from to collect data from it.
"""
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = 'euler'

# Angle at which to fail the episode
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4

# Angle limit set to 2 * theta_threshold_radians so failing observation
# is still within bounds.
high = np.array([x_threshold * 2,
                    np.finfo(np.float32).max,
                    theta_threshold_radians * 2,
                    np.finfo(np.float32).max],
                dtype=np.float32)

def cartpoleReward(state, action):
    x, x_dot, theta, theta_dot = state

    force = force_mag if action >= 0.5 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if kinematics_integrator == 'euler':
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot

    # done = bool(
    #     x < -x_threshold
    #     or x > x_threshold
    #     or theta < -theta_threshold_radians
    #     or theta > theta_threshold_radians
    # )

    reward = (1 - (x ** 2) / 11.52 - (theta ** 2) / 288)
    return reward
"""
The important line from the above is the asignment
$$reward = 1 - \frac{x^2}{11.52} - \frac{\theta^2}{288} = 1 - \frac{1}{2}\left(\frac{x}{2.4}\right)^2 - \frac{1}{2}\left(\frac{\theta}{12}\right)^2 $$

The above takes 1 and subtracts the simple average of the normalized squared position and angle. We can see that with an increase in the absolute values of x and θ, the reward decreases and reaches 0 when |x| = 2.4 and |θ| = 12. Note that the angle and position in this reward function are functions themselves of the current action and previous state (angle, position, velocity, and angle velocity).
"""

## Setup for Algorithms
"""
The following cells will guide through the setup of the variables necessary to run through the three algorithms.
We used the Numba package in order to heavily reduce the runtime of various functions
"""
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
X = (np.load('random-agent/random-cartpole-states.npy'))[:5000].T # states
U = (np.load('random-agent/random-cartpole-actions.npy'))[:5000].T # actions
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

## Algorithm 1
"""
We implemented the learning algorithm as outlined in our Koopman RL write-up

The optimal value function $V$ satisfies the (regularized) Hamilton-Jacobi-Bellman (HJB) equation
\begin{align}
    \rho v(x) &= \sup_{\pi_t \in \mathcal{P}(U)} \int_U \left( r(x,u) - \lambda \ln{\pi_t(u)} + (\mathcal{L}v)(x,u)\right) \pi_t(u) \text{d}u
\end{align}
where $\mathcal{P}(U) := \{ \pi_t:\int_U \pi_t(u) \text{d}u = 1 \text{ and } \pi_t(u) \geq 0 \text{ a.e on } U \}$. Using \eqref{est_eigen} applied to the value function $v$, we plug in our estimated 
eigenfunctions and also reduce the dimension by choosing a cut off $c<n$ to give us an approximate characterization of optimality which we call {\bf Koopman HJB}:

\begin{align}
    \rho v(x) &= \sup_{\pi_t \in \mathcal{P}(U)} \int_U \left (r(x,u) - \lambda \ln{\pi_t(u)}\right)\pi_t(u) \text{d}u + \int_U \sum_{\ell = 1}^c \langle \widehat{\varphi}_\ell, v \rangle \widehat \varphi_\ell(x,u) \pi_t(u) \text{d}u \label{inner_prod_approx}
    \\
    &= \sup_{\pi_t \in \mathcal{P}(U)} \int_U \left(r(x,u) - \lambda \ln{\pi_t(u)} \right)\pi_t(u)\text{d}u + \sum_{\ell = 1}^cm^v_\ell\lambda_\ell\int_U   \widehat{\varphi}_\ell(x,u)  \pi_t(u) \text{d}u \label{KoopmanHJB}
\end{align}

%If r(x,u) is also sufficient smooth, and the state-actions which show up in the cost function are themselves eigenfunctions of the Koopman operator (i.e. they are martingales) we can reformulate the reward function as a quadratic in terms of the vector of eigenfunctions $\Phi$:

Solving this maximization policy we get the feedback control:
\begin{align}
    \pi^*(u|x) &= \frac{\exp\left(\frac{1}{\lambda}(r(x,u) + (\mathcal{L}v)(x,u))\right)}{\int_U \exp\left(\frac{1}{\lambda}(r(x,u) + (\mathcal{L}v)(x,u))\right)du}\notag
    \\
    &\approx \frac{\exp\left(\frac{1}{\lambda}(r(x,u) + \sum_{\ell = 1}^c m^v_\ell\lambda_\ell \widehat{\varphi}_\ell(x,u))  \right)}{\int_U \exp\left(\frac{1}{\lambda}(r(x,u) + \sum_{\ell = 1}^c m^v_\ell\lambda_\ell \widehat{\varphi}_\ell(x,u))  \right)du} \label{approxOptPolicy}
    \\
    &=: \widehat{\pi}^*(u | x, v)
\end{align}





To run our algorithm, we initialize V in either some random way or we set it to 0. If we have some informed prior of what the form of the value function is, for example, using the assumption that the optimal value function is in the span of the dictionary space, we can probably speed up convergence significantly. 

Next, we want to find the OLS projection matrix B^\top_v of V^{\pi_0^*} onto the dictionary space by solving the following least squares problem 
\begin{align}
   \min_{B_g}\; \lVert G_{\tilde X} - B_g^\top \Psi_{\tilde X}\rVert_F \label{gProjPsi}
\end{align}

Next, We approximate \mathcal{L}v in 
\begin{align}
    (\mathcal{L}g)(\tilde{x})  \approx \sum_{\ell = 1}^n \lambda_\ell \widehat{\varphi}_\ell(\tilde{x}) m^g_\ell \label{est_eigen}
\end{align}
which we then plug into 
\begin{align}
    \frac{\exp\left(\frac{1}{\lambda}(r(x,u) + (\mathcal{L}v)(x,u))\right)}{\int_U \exp\left(\frac{1}{\lambda}(r(x,u) + (\mathcal{L}v)(x,u))\right)du}
\end{align}
to get our estimated optimal policy \hat{pi}^*(u | x, v)

Once we have our updated \hat{pi}^*(u | x, v), we can plug that into
\begin{align}
    \sup_{\pi_t \in \mathcal{P}(U)} \int_U \left(r(x,u) - \lambda \ln{\pi_t(u)} \right)\pi_t(u)\text{d}u + \sum_{\ell = 1}^cm^v_\ell\lambda_\ell\int_U   \widehat{\varphi}_\ell(x,u)  \pi_t(u) \text{d}u \label{KoopmanHJB}
\end{align}
to get our updated V^{\pi_j^*}

We repeat this process for t timesteps, specified by the caller of the function.
"""
def learningAlgorithm(L, X, Psi_X_tilde, U, reward, timesteps=100, cutoff=8, lamb=0.05):
    # placeholder functions
    V = lambda x: x
    pi_hat_star = lambda x: x

    low = np.min(U)
    high = np.max(U)

    constant = 1/lamb

    eigenvalues, eigenvectors = sp.linalg.eig(L)
    eigenvectors = eigenvectors
    @nb.njit(fastmath=True)
    def eigenfunctions(i, psi_x_tilde):
        return np.dot(np.real(eigenvectors[i]), psi_x_tilde) #Psi_X_tilde[:, l]

    eigenvectors_inverse_transpose = sp.linalg.inv(eigenvectors).T

    currentV = np.zeros(X.shape[1]) # V^{\pi*_0}
    lastV = currentV.copy()
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

"""
Unfortunately the results are a little strange. For one, the algorithm far too computationally complex, resulting in roughly a few minutes of compute per timestep.
The other issue we were finding is that the optimal policy, regardless of the state, always has a slight preference to pick action 0 over action 1 (1.02 vs 0.98).
This of course is incorrect as we would expect that the policy would prefer action 1 in the case where action 0 would cause you to terminate the episode.
"""

## Algorithm 2
"""
Sinha et al. (https://arxiv.org/pdf/1909.12520.pdf) proposed an algorithm, termed Recursive EDMD, for learnign the Koopman operator in an online learning setting.
We propose an altered algorithm that allows us to retrieve the Koopman generator \mathcal{L} by making use of gEDMD in order to get its eigenfunctions and running the rest of the algorithm.
"""
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

## Algorithm 3
"""
Now that we can calculate the Koopman generator in an online learning setting by calling rgEDMD
every time a new observation is made, we can, by extension, also run the learning algorithm in an online learning setting.
The following is the combination of the two algorithms.
"""

def onlineKoopmanLearning(X_tilde, Psi_X_tilde, dPsi_X_tilde):
    X_tilde_builder = X_tilde[:,:2]
    Psi_X_tilde_builder = Psi_X_tilde[:,:2]
    dPsi_X_tilde_builder = dPsi_X_tilde[:,:2]
    k = dPsi_X_tilde_builder.shape[0]

    z_m = np.zeros((k,k))
    phi_m_inverse = np.linalg.inv(np.identity(k))
    for x_tilde in X_tilde.T: # for each data point
        dPsi_X_tilde, z_m, phi_m_inverse, L_m = rgEDMD(
            x_tilde, X_tilde_builder, Psi_X_tilde_builder, dPsi_X_tilde_builder, k, z_m, phi_m_inverse
        ) # add new data point
        _, pi = learningAlgorithm(L, X, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=2, lamb=1) # learn new optimal policy

    # _, pi = learningAlgorithm(L, X, Psi_X_tilde, np.array([0,1]), cartpoleReward, timesteps=2, lamb=1)
    return pi # esimated optimal policy

"""
The way the algorithm is written, it might be infeasible to test since it has to run the computationally expensive learning algorithm every time a new data point is added.
For testing purposes we will uncomment the line before "return pi" and comment out the looping learningAlgorithm call.
"""

## Theoretical Considerations

### TODO: Closure of Iteration Procedure in the Dictionary Space
'''
Here we would like to show that from one iteration to the next the updated value function still lies in the span of the dictionary functions. Let $H = span(\psi)$ where we recall that $\psi = (\psi_1,...,\psi_k)^\top$. We first start with a value function $V^{\pi^*_j}$ which we assume, along with the reward function $r(x,u)$ to be in $H$ as part of our induction assumption. This implies from \eqref{approxOptPolicy} that we have
\begin{align}
    \widehat{\pi}_{j+1}^*(u | x, v) = \frac{\exp\left(\sum_{\ell = 1}^c \alpha_{j,\ell}\widehat{\varphi}_\ell(x,u)\right)  }{\int_U \exp\left(\sum_{\ell = 1}^c \alpha_{j,\ell}\widehat{\varphi}_\ell(x,u)\right) du} 
\end{align}

Plugging this into \eqref{KoopmanHJB} to get $V^{\pi^*_{j+1}}$, it is unclear if $V^{\pi^*_{j+1}}\in H$, i.e. that the new value function remains in the span of the dictionary functions.
'''
'''
### TODO: Operator Algebra Approach
Let $\mathcal{E}_x$ represent the conditional expectation over $\pi^*(\cdot|x)$, then we can represent the HJB expression as 
\begin{align}
    \rho V  &= \mathcal{E}_x r - \mathcal{E}^*_x \ln \pi_t
    \\
    \implies (\rho I - \mathcal{E}_x \mathcal{L})V &= \mathcal{E}_x(r - \ln \pi_t)
    \\
    \implies \ln \pi^* &= \left(\rho\mathcal{E}^{-1}_x -\mathcal{L}\right)V....
\end{align}

Along the lines of operator analysis, as discussed with Wen, we would like to show that the overall procedure of projecting each iteration of the value function on the estimated eigensystem, finding the estimated optimal policy, and then finding the new value function results in a contraction map. If we assume that the optimal value function itself lies in the span of the dictionary space, it seems intuitive that this proceedure should converge to the optimal value function since each iteration some kind of composition between a projection operator and a Bellman operator, both of which are contractive. 
'''



'''
# TODO: Mean Field Game Approaches Using Stochastic Maximum Principle
There seem to be some promising works at the intersection of relaxed control theory and mean field games. Mixed strategies look very close to MDP problems and the way that MFGs are sometimes solved is with Pontryagin's maximum principle. See Daneil Lacker's thesis and IPAM summary papers (\href{http://www.columbia.edu/~dl3133/dlacker-dissertation.pdf}{Thesis Link}, \href{http://www.columbia.edu/~dl3133/IPAM-MFGCompactnessMethods.pdf}{IPAM Lecture Link}, \href{http://www.ipam.ucla.edu/programs/summer-schools/graduate-summer-school-mean-field-games-and-applications/?tab=schedule}{IPAM Lecture Video})
'''