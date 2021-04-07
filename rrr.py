#%%
from base import *
import jax.scipy as jsp

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)
dPsi_X_T = dPsi_X.T

#%%
rank = 8 # arbitrarily selected
reg = 0 # regularization param

#%%
# To achieve (231, 231) shape:
# X = Psi_X.T
# Y = dPsi_X.T
calc_part_1 = sp.linalg.inv(Psi_X @ Psi_X_T) # pseudoinv?
calc_part_2 = Psi_X @ dPsi_X_T
B_ols = calc_part_1 @ calc_part_2
U, S, V = sp.linalg.svd(dPsi_X @ Psi_X_T @ B_ols)
W = V[0:rank].T

#%%
B_rr = B_ols @ W @ W.T

# %%
# B_rr is M
# M.T is L
L = B_rr.T
print(L.shape) # shape was (10000, 10000)
# %%
