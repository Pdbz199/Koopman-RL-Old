#%%
from base import *

#%%
# Construct \text{d}\Psi_X matrix
dPsi_X = np.empty((k, m))
for column in range(m-1):
    dPsi_X[:, column] = vectorized_dpsi(np.arange(k), column)

#%%
def sparsifyDynamics(Theta, dXdt, lamb, n):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(n):                       # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
            
    return Xi

#%%
lamb = 0.05 # sparsification knob lambda
Xi = sparsifyDynamics(Psi_X.T, dPsi_X.T, lamb, d)
#%%
L = Xi # estimate of Koopman generator

#%%
L_times_B_transposed = (L @ B).T
def b(l):
    return L_times_B_transposed @ Psi_X[:, l] # (k,)

#%% Performs really well!
r = 10
for l in range(r):
    b_l = b(l)
    print(f"b({r}):", b_l)

# %%
L_times_second_order_B_transpose = (L @ second_order_B).T
def a(l):
    return (L_times_second_order_B_transpose @ Psi_X[:, l]) - \
        (second_order_B.T @ nablaPsi[:, :, l] @ b(l))

#%% Reshape a vector as matrix and perform some tests
def covarianceMatrix(a_func, l):
    a_l = a_func(l)
    covariance = np.zeros((d, d))
    covariance[0, 0] = a_l[0]
    row = 0
    col = 1
    n = 1
    while col < d:
        covariance[row, col] = a_l[n]
        covariance[col, row] = a_l[n]
        if row == col: 
            col += 1
            row = 0
        else:
            row += 1
        n +=1
    return covariance

#%% Did not perform well ):
test = covarianceMatrix(a, 2)
test_df = pd.DataFrame(test)
print("a:", test_df)
print("a diagonal:", np.diagonal(test))

#%% Condition numbers - we want small numbers
print(np.linalg.cond(L)) # inf
print(np.linalg.cond(Psi_X)) # 98 million+
# %%