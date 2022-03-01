import numpy as np

import sys
sys.path.append('../../')
import estimate_L
import observables

def checkMatrixRank(X, name):
    if np.linalg.matrix_rank(X) != X.shape[0]:
        raise ValueError(f"{name} matrix is not full rank")

def checkConditionNumber(X, name, threshold=200):
    if np.linalg.cond(X) > threshold:
        raise ValueError(f"Condition number of {name} is too large")

class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi=observables.monomials(2),
        psi=observables.monomials(2),
        regressor='ols',
        p_inv=True
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
        self.Psi_U = self.psi(U)

        # Get dimensions
        self.dim_phi = self.Phi_X.shape[0]
        self.dim_psi = self.Psi_U.shape[0]

        # Make sure data is full rank
        checkMatrixRank(self.Phi_X, "Phi_X")
        checkMatrixRank(self.Phi_Y, "Phi_Y")
        checkMatrixRank(self.Psi_U, "Psi_U")

        # Make sure condition numbers are small
        checkConditionNumber(self.Phi_X, "Phi_X")
        checkConditionNumber(self.Phi_Y, "Phi_Y")
        checkConditionNumber(self.Psi_U, "Psi_U")

        # Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
        self.kronMatrix = np.empty([
            self.dim_psi * self.dim_phi,
            self.N
        ])
        for i in range(self.N):
            self.kronMatrix[:,i] = np.kron(
                self.Psi_U[:,i],
                self.Phi_X[:,i]
            )

        # Solve for M and B
        if regressor == 'rrr':
            self.M = estimate_L.rrr(self.kronMatrix.T, self.Phi_Y.T).T
            self.B = estimate_L.rrr(self.Phi_X.T, self.X.T)
        if regressor == 'sindy':
            self.M = estimate_L.SINDy(self.kronMatrix.T, self.Phi_Y.T).T
            self.B = estimate_L.SINDy(self.Phi_X.T, self.X.T)
        else:
            self.M = estimate_L.ols(self.kronMatrix.T, self.Phi_Y.T, p_inv).T
            self.B = estimate_L.ols(self.Phi_X.T, self.X.T, p_inv)

        # reshape M into tensor K
        self.K = np.empty([
            self.dim_phi,
            self.dim_phi,
            self.dim_psi
        ])
        for i in range(self.dim_phi):
            self.K[i] = self.M[i].reshape(
                [self.dim_phi, self.dim_psi],
                order='F'
            )

    def K_(self, u):
        ''' Pick out Koopman operator given an action '''

        # If array, convert to column vector
        if len(u.shape) == 1:
            u = np.vstack(u)
        
        K_u = np.einsum('ijz,zk->kij', self.K, self.psi(u))

        if K_u.shape[0] == 1:
            return K_u[0]

        return K_u