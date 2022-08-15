#%%
from binascii import Error
import numpy as np
import scipy as sp
import numba as nb

def checkMatrixRank(X, name):
    rank = np.linalg.matrix_rank(X)
    print(f"{name} matrix rank: {rank}")
    if rank != X.shape[0]:
        # raise ValueError(f"{name} matrix is not full rank ({rank} / {X.shape[0]})")
        pass

def checkConditionNumber(X, name, threshold=200):
    cond_num = np.linalg.cond(X)
    print(f"Condition number of {name}: {cond_num}")
    if cond_num > threshold:
        # raise ValueError(f"Condition number of {name} is too large ({cond_num} > {threshold})")
        pass

def gedmd(X, Y, rank=8):
    U, Sigma, VT = sp.linalg.svd(X, full_matrices=False)
    U_tilde = U[:, :rank]
    Sigma_tilde = np.diag(Sigma[:rank])
    VT_tilde = VT[:rank]

    M_tilde = sp.linalg.solve(Sigma_tilde.T, (U_tilde.T @ Y @ VT_tilde.T).T).T
    L = M_tilde.T # estimate of Koopman generator
    return L

#%% (Theta=Psi_X_T, dXdt=dPsi_X_T, lamb=0.05, n=d)
def SINDy(Theta, dXdt, lamb=0.05):
    d = dXdt.shape[1]
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10): #which parameter should we be tuning here for RRR comp
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(d):                       # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
            
    L = Xi
    return L

def ols(X, Y, pinv=True):
    if pinv:
        return np.linalg.pinv(X.T @ X) @ X.T @ Y
    return np.linalg.inv(X.T @ X) @ X.T @ Y

def OLS(X, Y, pinv=True):
    return ols(X, Y, pinv)

def rrr(X, Y, rank=8):
    B_ols = ols(X, Y) # if infeasible use GD (numpy CG)
    U, S, V = np.linalg.svd(Y.T @ X @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

def RRR(X, Y, rank=8):
    return rrr(X, Y, rank)

@nb.njit(fastmath=True)
def ridgeRegression(X, y, lamb=0.05):
    return np.linalg.inv(X.T @ X + (lamb * np.identity(X.shape[1]))) @ X.T @ y

class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi,
        psi,
        regressor='ols',
        p_inv=True,
        rank=8,
        is_generator=False,
        dt = 0.01
    ):
        self.X = X
        self.Y = Y
        self.U = U
        self.phi = phi
        self.psi = psi

        # Get number of data points
        self.N = self.X.shape[1]

        # Construct Phi and Psi matrices
        self.Phi_X = self.phi(X)
        self.Phi_Y = self.phi(Y)
        if is_generator:
            self.dt = dt
            self.regression_Y = (self.Phi_Y - self.Phi_X) / self.dt
        else:
            self.regression_Y = self.Phi_Y
        self.Psi_U = self.psi(U)

        # Get dimensions
        self.x_dim = self.X.shape[0]
        self.u_dim = self.U.shape[0]
        self.phi_dim = self.Phi_X.shape[0]
        self.psi_dim = self.Psi_U.shape[0]
        self.x_column_dim = [self.x_dim, 1]
        self.u_column_dim = [self.u_dim, 1]
        self.phi_column_dim = [self.phi_dim, 1]

        # Make sure data is full rank
        checkMatrixRank(self.Phi_X, "Phi_X")
        checkMatrixRank(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkMatrixRank(self.Psi_U, "Psi_U")

        # Make sure condition numbers are small
        checkConditionNumber(self.Phi_X, "Phi_X")
        checkConditionNumber(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkConditionNumber(self.Psi_U, "Psi_U")

        # Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
        self.kron_matrix = np.empty([
            self.psi_dim * self.phi_dim,
            self.N
        ])
        for i in range(self.N):
            self.kron_matrix[:,i] = np.kron(
                self.Psi_U[:,i],
                self.Phi_X[:,i]
            )

        # Solve for M and B
        if regressor.lower() == 'rrr':
            self.M = rrr(self.kron_matrix.T, self.regression_Y.T, rank).T
            self.B = rrr(self.Phi_X.T, self.X.T, rank)
        elif regressor.lower() == 'sindy':
            self.M = SINDy(self.kron_matrix.T, self.regression_Y.T).T
            self.B = SINDy(self.Phi_X.T, self.X.T)
        elif regressor.lower() == 'ols':
            self.M = ols(self.kron_matrix.T, self.regression_Y.T, p_inv).T
            self.B = ols(self.Phi_X.T, self.X.T, p_inv)
        else:
            raise Error("Did not pick a supported regression algorithm.")

        # reshape M into tensor K
        self.K = np.empty([
            self.phi_dim,
            self.phi_dim,
            self.psi_dim
        ])
        for i in range(self.phi_dim):
            self.K[i] = self.M[i].reshape(
                [self.phi_dim, self.psi_dim],
                order='F'
            )

    def K_(self, u):
        ''' Pick out Koopman operator given an action '''

        # If array, convert to column vector
        if isinstance(u, int) or isinstance(u, float) or isinstance(u, np.int64) or isinstance(u, np.float64):
            u = np.array([[u]])
        elif len(u.shape) == 1:
            u = np.vstack(u)
        
        K_u = np.einsum('ijz,zk->kij', self.K, self.psi(u))

        if K_u.shape[0] == 1:
            return K_u[0]

        return K_u

    def phi_f(self, x, u):
        """
            INPUTS:
                x - state column vector(s)
                u - action column vector(s)

            OUTPUTS:
                phi(x) column vector(s)
        """
        
        return self.K_(u) @ self.phi(x)

    def f(self, x, u):
        """
            INPUTS:
                x - state column vector(s)
                u - action column vector(s)

            OUTPUTS:
                x column vector(s)
        """

        return self.B.T @ self.phi_f(x, u)