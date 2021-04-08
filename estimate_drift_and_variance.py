#%%
from base import np, sp, d, m, k, Psi_X, nablaPsi, B, second_order_B

#%%
class EstimateDandV():
    def __init__(self, L):
        self.L = L # Koopman generator
        self.L_times_B_transposed = (L @ B).T
        self.L_times_second_order_B_transpose = (L @ second_order_B).T
        # Eigendecomposition of Koopman generator
        self.eigenvalues, self.eigenvectors = sp.linalg.eig(L)
        # Get eigenfunctions (can this be reduced?)
        self.eigenfunctions = self.eigenvectors.T @ Psi_X
        # Calculate Koopman (generator?) modes
        self.V_v1 = B.T @ sp.linalg.inv(self.eigenvectors.T)
        self.V_v2 = second_order_B.T @ sp.linalg.inv(self.eigenvectors.T)

    """====================== VERSION 1 ======================"""
    # Computed b function (sometimes denoted by \mu) without dimension reduction
    def b(self, l):
        return self.L_times_B_transposed @ Psi_X[:, l] # (k,)

    def a(self, l):
        return (self.L_times_second_order_B_transpose @ Psi_X[:, l]) - \
            (second_order_B.T @ nablaPsi[:, :, l] @ self.b(l))

    """====================== VERSION 2 ======================"""
    # The b_v2 function allows for heavy dimension reduction
    # default is reducing by 90% (taking the first k/10 eigen-parts)
    # TODO: Figure out correct place to take reals
    def b_v2(self, l, num_dims=k//10, V=self.V_v1):
        res = 0
        for ell in range(k-1, k-num_dims, -1):
            res += self.eigenvalues[ell] * self.eigenfunctions[ell, l] * V[:, ell] #.reshape(-1, 1)
        return np.real(res)

    def a_v2(self, l):
        return (self.b_v2(l, V=self.V_v2)) - \
            (second_order_B.T @ nablaPsi[:, :, l] @ self.b_v2(l))

    """====================== VERSION 3 ======================"""
    def b_v3(self):
        B = np.zeros((d, m))
        m_range = np.arange(m)
        B = X[:, m_range] - Z[:, m_range]
        print("B shape:", B.shape)
        print("Psi_X transpose shape:", Psi_X_T.shape)
        PsiMult = sp.linalg.inv(Psi_X @ Psi_X_T) @ Psi_X
        C = PsiMult @ B.T
        # Each col of matric C represents the coeficients in a linear combo of the dictionary functions that makes up each component of the drift vector. So each c_{} 
        print("C shape:", C.shape)

        b_v3 = C.T @ Psi_X
        return b_v3

    def a_v3(self, l):
        diffusionDictCoefs = np.empty((d, d, k))
        diffusionMat = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                Bij = B[i] * B[j]
                diffusionDictCoefs[i, j] = PsiMult @ Bij
                diffusionMat[i, j] = np.dot(diffusionDictCoefs[i, j], Psi_X[:,l])
        return diffusionMat