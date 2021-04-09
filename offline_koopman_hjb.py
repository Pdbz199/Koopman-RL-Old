#%%
from base import * #np, sp, n, B
from estimate_L import *

#%%
def ln(x):
    return np.log(x)

#%%
# X_tilde = np.append(X, U, axis=0) # already computed for L
# Psi_X_tilde = psi(X_tilde) # already computed for L
def learningAlgorithm(L, X, Psi_X_tilde, U, reward, cutoff=8, lamb=0.05, epsilon=0.1):
    low = np.min(U)
    high = np.max(U)

    constant = (1/lamb)

    eigenvalues, eigenvectors = sp.linalg.eig(L) # L created with X_tilde
    eigenfunctions = lambda ell, l: np.dot(eigenvectors[ell], Psi_X_tilde[:, l])

    eigenvectors_inverse_transpose = sp.linalg.inv(eigenvectors).T # pseudoinverse?

    # j = 1 # TODO: is this j index useful?
    V = np.zeros(X.shape[1]) # V^{\pi*_0}
    lastV = V + (epsilon+0.1)

    while (abs(V - lastV) > epsilon).any(): # there may be a more efficient way with maintaining max
        G_X_tilde = V.copy()
        B_g = rrr(Psi_X_tilde.T, G_X_tilde.T)

        generatorModes = B_g.T @ eigenvectors_inverse_transpose

        def Lv_hat(l):
            summation = 0
            for ell in range(cutoff):
                summation += eigenvalues[ell] * eigenfunctions(ell, l) * generatorModes[ell]
            return summation

        compute = lambda u, l: np.exp(constant * (reward(X[:,l], u) + Lv_hat(l)))

        def pi_hat_star(u, l): # action given state
            numerator = compute(u, l)
            denominator = sp.integrate.quad(compute, low, high, args=(l,))
            return numerator / denominator
        
        def integral_summation(l):
            for ell in range(cutoff):
                generatorModes[ell] * eigenvalues[ell] * \
                    sp.integrate.quad(
                        lambda u, l: eigenfunctions(ell, np.append(X[:,l], u, axis=0)) * pi_hat_star(u, l),
                        low, high, args=(l,)
                    )

        def V(l):
            return sp.integrate.quad(
                lambda u, l: (reward(X[:,l], u) - (lamb * ln(pi_hat_star(u, l)))) * pi_hat_star(u, l),
                low, high, args=(l,)
            ) + integral_summation(l)

        lastV = V
        for i in range(V.shape[0]):
            V[i] = V(i)

        # j+=1

# %%
