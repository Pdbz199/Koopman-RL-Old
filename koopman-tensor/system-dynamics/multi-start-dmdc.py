#%%
import numpy as np
np.random.seed(123)
import sys
sys.path.append('../../')
import estimate_L
import observables

A = np.array([
    [1.5, 0],
    [0, 0.1]
])
B = np.array([
    [1],
    [0]
])

def F(x,u):
    return A @ x + B @ [u]

action_bounds = [10.1, -10.0]
def random_control(x):
    ''' compute random control variable u '''
    u = np.random.uniform(high=action_bounds[0], low=action_bounds[1])
    return u
def nonrandom_control(x):
    ''' compute state-dependent control variable u '''
    u = -1*x[0] #+ np.random.randn()
    return u
def pi_train(x):
    return 1 * x[0]
def pi_test(x):
    return -1 * x[0] - np.random.randn()

action_grid = np.array([i/10 for i in range(-100,101,1)])
def nearest_action(u):
    if u > 10.0: return 10.0
    if u < -10.0: return -10.0
    return np.round(u, 1)
def action_to_index(u):
    return int(np.round((nearest_action(u) - action_bounds[1]) / 0.1))
def index_to_action(i):
    return i*0.1 + action_bounds[1]
split_datasets = {}
for action in action_grid:
    split_datasets[action] = {"x":[],"x_prime":[]}

# def phi(x):
#     ''' Identity for DMD '''
#     return x

# def psi(u):
#     ''' Identity for DMD '''
#     return u

monomials = observables.monomials(2)

def phi(x):
    ''' Quadratic dictionary '''
    if len(x.shape) == 1:
        return monomials(x.reshape(-1,1))[:,0]
    if x.shape[0] == 1 or x.shape[1] == 0:
        return monomials(x)[:,0]
    return monomials(x)

def psi(u):
    ''' Quadratic dictionary '''
    return phi(u)

#%% simulate system to generate data matrices
m = 10000 #500 # number of sample steps from the system.
n = 2 # dimensionality of state space
q = 1 # dimensionality of control space

#%% Control snapshotting
X = np.random.normal(loc=8, scale=5, size=(n, m*2-1))
# X = np.full((n, m*2-1), np.array([[4],[7]]))
# X[:,0] = np.array([6,16])
Y = np.empty((n, m*2-1))
U = np.empty((q, m*2-1))
# sys = UnstableSystem1(x0)
for k in range(m*2-1):
    # separate state and action
    u_k = pi_train(X[:, k]) #random_control(X[:, k])
    U[:, k] = u_k
    y = F(X[:, k], u_k)
    Y[:, k] = np.squeeze(y)
    split_datasets[nearest_action(u_k)]['x'].append(X[:,k])
    split_datasets[nearest_action(u_k)]['x_prime'].append(Y[:,k])

#%% Build Phi and Psi matrices
d_phi = 6
d_psi = 3 
N = m*2-1

Phi_X = np.empty((d_phi, N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(x)

Phi_Y = np.empty((d_phi, N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(y)

for action in split_datasets:
    split_datasets[action]['x'] = np.array(split_datasets[action]['x']).T
    split_datasets[action]['x_prime'] = np.array(split_datasets[action]['x_prime']).T

    split_datasets[action]['phi_x'] = phi(split_datasets[action]['x'])
    split_datasets[action]['phi_x_prime'] = phi(split_datasets[action]['x_prime'])

Psi_U = np.empty((d_psi, N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(u)

# def phi(x):
#     return np.array([1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2, x[2], x[0]*x[2], x[1]*x[2], x[2]**2])

XU = np.append(X, U, axis=0)
d_phi_xu = phi(XU[:,0]).shape[0] # 6 + (3-1) + 2 = 10
Phi_XU = np.empty((d_phi_xu, N))
for i,xu in enumerate(XU.T):
    Phi_XU[:,i] = phi(xu)

Phi_XU_prime = Phi_XU[:,1:]
Phi_XU = Phi_XU[:,:-1]

#%% Concatenated state and action Koopman operator
Koopman_operator = estimate_L.ols(Phi_XU.T, Phi_XU_prime.T).T
fun_koopman_operator = estimate_L.ols(Phi_XU.T, Y[:,:-1].T).T

#%% Separate Koopman operator per action
Koopman_operators = [] # np.empty((action_grid.shape[0],d_phi,d_phi))
for action in split_datasets:
    if np.array_equal(split_datasets[action]['phi_x'], [1]):
        Koopman_operators.append(np.zeros((d_phi,d_phi)))
        continue

    Koopman_operators.append(estimate_L.ols(split_datasets[action]['phi_x'].T, split_datasets[action]['phi_x_prime'].T).T)
Koopman_operators = np.array(Koopman_operators, dtype=object)

#%% Build kronMatrix
kronMatrix = np.empty((d_psi * d_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Y.T).T # Phi_Y.T

#%% Reshape M into K tensor
K = np.empty((2, d_phi, d_psi))
for i in range(2):
    K[i] = M[i].reshape((d_phi,d_psi), order='F')

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(u))

#%% Training error
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

fun_norms = []
concatenated_norms = []
norms = []
for i in range(N-1):
    true_phi_x_prime = Y[:,i]
    predicted_phi_xu_prime = fun_koopman_operator @ Phi_XU[:,i]
    fun_norms.append(l2_norm(true_phi_x_prime, predicted_phi_xu_prime))

    # Concatenated
    true_phi_xu_prime = Phi_XU_prime[:,i]
    predicted_phi_xu_prime = Koopman_operator @ Phi_XU[:,i]
    concatenated_norms.append(l2_norm(true_phi_xu_prime, predicted_phi_xu_prime))

    # Tensor
    true_phi_x_prime = Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
fun_norms = np.array(fun_norms)
concatenated_norms = np.array(concatenated_norms)
norms = np.array(norms)

# Split datasets
split_datasets_norms = []
for action in split_datasets:
    if np.array_equal(split_datasets[action]['phi_x'], [1]):
        continue

    for i in range(split_datasets[action]['phi_x_prime'].shape[1]):
        true_phi_x_prime = split_datasets[action]['phi_x_prime'][:,i]
        predicted_phi_x_prime = Koopman_operators[action_to_index(action)] @ split_datasets[action]['phi_x'][:,i]
        split_datasets_norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
split_datasets_norms = np.array(split_datasets_norms)

print("Fun mean norm on training data:", fun_norms.mean())
print("Concatenated mean norm on training data:", concatenated_norms.mean())
print("Split datasets mean norm on training data:", split_datasets_norms.mean())
print("Tensor mean norm on training data:", norms.mean())





#%% Reset split datasets
split_datasets = {}
for action in action_grid:
    split_datasets[action] = {"x":[],"x_prime":[]}

#%% State snapshotting
snapshots = np.empty((n, m))
snapshots[:, 0] = np.array([16,10])

#%% Control snapshotting
U = np.empty((q, m-1))
# sys = UnstableSystem1(x0)
for k in range(m-1):
    u_k = pi_test(snapshots[:, k])
    y = F(snapshots[:, k], u_k)
    snapshots[:, k+1] = np.squeeze(y)
    U[:, k] = u_k

X = snapshots[:, :m-1]
Y = snapshots[:, 1:m]

#%% Build Phi and Psi matrices
d_phi = 6
d_psi = 3
N = m-1

Phi_X = np.empty((d_phi, N))
for i,x in enumerate(X.T):
    Phi_X[:,i] = phi(x)

Phi_Y = np.empty((d_phi, N))
for i,y in enumerate(Y.T):
    Phi_Y[:,i] = phi(y)

Psi_U = np.empty((d_psi, N))
for i,u in enumerate(U.T):
    Psi_U[:,i] = psi(u)

XU = np.append(X, U, axis=0)
d_phi_xu = phi(XU[:,0]).shape[0]
Phi_XU = np.empty((d_phi_xu, N))
for i,xu in enumerate(XU.T):
    Phi_XU[:,i] = phi(xu)

#%% Prediction error
fun_norms = []
concatenated_norms = []
split_datasets_norms = []
norms = []
for i in range(N-1):
    true_phi_x_prime = Y[:,i]
    predicted_phi_xu_prime = fun_koopman_operator @ Phi_XU[:,i]
    fun_norms.append(l2_norm(true_phi_x_prime, predicted_phi_xu_prime))

    # Concatenated
    true_phi_xu_prime = Phi_XU[:,i+1]
    predicted_phi_xu_prime = Koopman_operator @ Phi_XU[:,i]
    concatenated_norms.append(l2_norm(true_phi_xu_prime, predicted_phi_xu_prime))

    # Split datasets
    true_phi_x_prime = Phi_Y[:,i]
    predicted_phi_x_prime = Koopman_operators[action_to_index(U[0,i])] @ Phi_X[:,i]
    split_datasets_norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))

    # Tensor
    true_phi_x_prime = Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
fun_norms = np.array(fun_norms)
concatenated_norms = np.array(concatenated_norms)
split_datasets_norms = np.array(split_datasets_norms)
norms = np.array(norms)

print("Fun mean norm on prediction data:", fun_norms.mean())
print("Concatenated mean norm on prediction data:", concatenated_norms.mean())
print("Split datasets mean norm on prediction data:", split_datasets_norms.mean())
print("Tensor mean norm on prediction data:", norms.mean())

#%%