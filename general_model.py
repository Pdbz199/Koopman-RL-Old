#%%
import observables
import numpy as np
import scipy as sp
import numba as nb
import estimate_L
from scipy import integrate
from algorithms import learningAlgorithm, rgEDMD#, onlineKoopmanLearning

# @nb.njit(fastmath=True)
# def ln(x):
#     return np.log(x)

@nb.njit(fastmath=True) #, parallel=True)
def nb_einsum(A, B):
    assert A.shape == B.shape
    res = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res += A[i,j]*B[i,j]
    return res

@nb.njit(fastmath=True)
def dpsi(X, nablaPsi, nabla2Psi, k, l, t=1):
    difference = X[:, l+1] - X[:, l]
    term_1 = (1/t) * (difference)
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * np.outer(difference, difference)
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1, term_2) + nb_einsum(term_3, term_4)

def rejection_sampler(p, xbounds, pmax):
    while True:
        x = np.random.rand(1)*(xbounds[1]-xbounds[0])+xbounds[0]
        y = np.random.rand(1)*pmax
        if y<=p(x[0]):
            return x

def sample_cartpole_action(pi, x):
    f1 = pi(0.2, x)
    f2 = pi(0.7, x)
    y = np.random.rand()
    if y <= 0.5*f1 and y >= 0:
        return y / f1
    
    return (y - 0.5 * f1 + 0.5 * f2) / f2

class GeneratorModel:
    def __init__(self, psi, reward):
        """
        Create instance of model

            Parameters:
                psi: Set of dictionary functions from the observables class
                reward: The reward function for whatever problem you are trying to model
        """
        self.psi = psi
        self.reward = reward

    def fit(self, X, U):
        """
        Fits a policy pi to the dataset using Koopman RL

            Parameters:
                X: State data
                U: Action data

        """
        self.X = X
        self.U = U
        self.min_action = np.min(U)
        self.max_action = np.max(U)

        self.X_tilde = np.append(X, [U], axis=0) # extended states
        self.d = self.X_tilde.shape[0]
        self.m = self.X_tilde.shape[1]
        # self.s = int(self.d*(self.d+1)/2) # number of second order poly terms
        
        self.Psi_X_tilde = self.psi(self.X_tilde)
        # self.Psi_X_tilde_T = Psi_X_tilde.T
        self.k = self.Psi_X_tilde.shape[0]
        self.nablaPsi = self.psi.diff(self.X_tilde)
        self.nabla2Psi = self.psi.ddiff(self.X_tilde)

        self.dPsi_X_tilde = np.zeros((self.k, self.m))
        for row in range(self.k):
            for column in range(self.m-1):
                self.dPsi_X_tilde[row, column] = dpsi(
                    self.X_tilde, self.nablaPsi,
                    self.nabla2Psi, row, column
                )
        # self.dPsi_X_tilde_T = dPsi_X_tilde.T

        # L = rrr(Psi_X_tilde_T, dPsi_X_tilde_T)
        self.L = estimate_L.rrr(self.Psi_X_tilde.T, self.dPsi_X_tilde.T)
        # self.L = estimate_L.rrr(self.Psi_X_tilde, self.dPsi_X_tilde)

        self.z_m = np.zeros((self.k, self.k))
        self.phi_m_inverse = np.linalg.inv(np.identity(self.k))

        self.V, self.pi = learningAlgorithm(
            self.L, self.X, self.Psi_X_tilde,
            self.U, self.reward, timesteps=2, lamb=0.5
        )

        self.pi_with_state = lambda u, x: self.pi(u, )

    def update(self, x, u):
        """
        Updates the model to include data about a new point (this assumes only two states/actions were given during the fitting process)

            Parameters:
                x: A single state vector
                u: A single action vector
        """
        self.dPsi_X_tilde, self.z_m, self.phi_m_inverse, self.L = rgEDMD(
            np.append(x, u),
            self.X_tilde,
            self.psi,
            self.Psi_X_tilde,
            dpsi,
            self.dPsi_X_tilde,
            self.k,
            self.z_m,
            self.phi_m_inverse
        )

        self.V, self.pi = learningAlgorithm(
            self.L, self.X, self.Psi_X_tilde,
            self.U, self.reward, timesteps=2, lamb=0.5
        )

    def sample_action(self, x):
        """
        Sample action from policy pi

            Parameters:
                x: Current state vector

            Returns:
                action: Action sampled from estimated optimal policy pi
        """
        return sample_cartpole_action(self.pi, x)

        # Attempt at taylor approximation sampling
        # pi_hat_star_hat = interpolate.approximate_taylor_polynomial(self.pi, np.mean(U), 2, 1)
        # rand = np.random.normal(loc=, scale=)
        # sampled_action = pi_hat_star_hat(rand)

        # try:
        #     return rejection_sampler(lambda u: self.pi(u, 100), [self.min_action,self.max_action], 1.1)[0]
        # except:
        #     self.V, self.pi = learningAlgorithm(
        #         self.L, self.X, self.Psi_X_tilde,
        #         self.U, self.reward, timesteps=2, lamb=0.5
        #     )
        
        # return rejection_sampler(lambda u: self.pi(u, 100), [self.min_action,self.max_action], 1.1)[0]
# %%
