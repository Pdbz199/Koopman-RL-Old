import numpy as np

import sys
sys.path.append('../')
import estimate_L
import observables

class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi=observables.monomials(2),
        psi=observables.monomials(2)
    ):
        self.X = X
        self.Y = Y
        self.U = U
        self.phi = phi
        self.psi = psi

        # Get number of data points
        self.N = self.X.shape[1]

        # Construct Phi and Psi matrices
        self.Phi_X = phi(X)
        self.Phi_Y = phi(Y)
        self.Psi_U = psi(U)

        self.dim_phi = self.Phi_X.shape[0]
        self.dim_psi = self.Psi_U.shape[0]

        # Build matrix of kronecker products between u_i and x_j for all i, j
        self.kronMatrix = np.empty([
            self.dim_phi * self.dim_psi,
            self.N
        ])
        for i in range(self.N):
            self.kronMatrix[:,i] = np.kron(
                self.Psi_U[:,i],
                self.Phi_X[:,i]
            )

        # Solve for M
        self.M = estimate_L.ols(self.kronMatrix.T, self.Phi_Y.T).T

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
        
        return np.einsum('ijz,zk->kij', self.K, self.psi(u))