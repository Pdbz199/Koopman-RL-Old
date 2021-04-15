#%%
from base import *

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for row in range(k):
    for column in range(m-1):
        dPsi_X[row, column] = dpsi(X, row, column)
dPsi_X_T = dPsi_X.T

#%%
rank = 8 # arbitrarily selected
reg = 0 # regularization param

#%%
# To achieve (231, 231) shape:
# X = Psi_X.T
# Y = dPsi_X.T
calc_part_1 = sp.linalg.pinv(Psi_X @ Psi_X_T) # pseudoinv?
calc_part_2 = Psi_X @ dPsi_X_T
B_ols = calc_part_1 @ calc_part_2
U, S, V = sp.linalg.svd(dPsi_X @ Psi_X_T @ B_ols)
W = V[0:rank].T

#%%
B_rr = B_ols @ W @ W.T

# %%
# B_rr is M
# M.T is L
L = B_rr
print(L.shape) # shape was (10000, 10000)

# %%
L_times_B_transposed = (L @ B).T
def b(l):
    return L_times_B_transposed @ Psi_X[:, l] # (k,)

#%% Really terrible
r = 10
for l in range(r):
    b_l = b(l)
    print(f"b({l}):", b_l)

# %%
L_times_second_order_B_transpose = (L @ second_order_B).T
def a(l):
    return (L_times_second_order_B_transpose @ Psi_X[:, l]) - \
        (second_order_B.T @ nablaPsi[:, :, l] @ b(l))

#%%
test = vectorToMatrix(a(2))
test_df = pd.DataFrame(test)
print("a:", test_df)
print("a diagonal:", np.diagonal(test))

#%%