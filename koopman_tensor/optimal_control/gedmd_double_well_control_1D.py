#%%
import numpy as np
np.random.seed(123)
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

import sys
sys.path.append('../../')
import algorithmsv2
#import DoubleWell_GenerateData
import domain
import estimate_L
import observables

def sortEig(A, evs=5):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    d, V = np.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind[:evs]], V[:, ind[:evs]])

class DoubleWell():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return -x**3 + 2*x + self.c
    
    def sigma(self, x):
        return np.sqrt(2/self.beta)

init_beta = 1
init_c = 0
s = DoubleWell(beta=init_beta, c=init_c)
h = 1e-2
y = 1

x = np.linspace(-2.5, 2.5, 1000)

# The parametrized function to be plotted
def f(x, beta, c):
    return 1/4*x**4 - x**2 - c*x


# Define initial parameters
u_bounds = np.array([[-2, 2]])
x_bounds = np.array([[-1.5, 1.5]])
#bounds_u = np.array([u_bounds])
u_boxes = np.array([20])
x_boxes = np.array([15])
Omega_u = domain.discretization(u_bounds, u_boxes)
Omega_x = domain.discretization(x_bounds, x_boxes)
# I think X and U need to be generated in a different way
X = Omega_x.randPerBox(100)
# U = np.random.uniform(u_bounds[0], u_bounds[1], (1, X.shape[1]))
U = Omega_u.randPerBox(100)

dim_x = X.shape[0]
dim_u = U.shape[0]
Y = s.b(X[:,0])
Z = s.sigma(X[:, 0])


#%% Define observables
order = 6
phi = observables.monomials(order)
psi = observables.monomials(order) #lambda u: np.array([1])

#%% Build Phi and Psi matrices
N = X.shape[1]
Phi_X = phi(X)
Psi_U = psi(U) #np.ones((1,N))
dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]

dPhi_Y = phi.diff(X)[:,0,:]*Y*h #np.einsum('ijk,jk->ik', phi.diff(X), Y)
ddPhi_X = phi.ddiff(X)*h/2 # second-order derivatives
#S = np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
S = Z * Z
for i in range(dim_phi):
    dPhi_Y[i, :] += 0.5*np.sum( ddPhi_X[i,:,:,:] * S, axis=(0,1) )

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, dPhi_Y.T).T

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    psi_u = psi(u.reshape(-1,1))[:,0]
    return np.einsum('ijz,z->ij', K, psi_u)

#%% Get eigenvalues/vectors
evs = 3
w, V = sortEig(K_u(K, np.array([0])).T, evs)

#%%
c = Omega_u.midpointGrid()
Phi_c = phi(c)
for i in range(evs):
    plt.figure(i+1)
    plt.clf()
    Omega_u.plot(np.real( V[:, i].T @ Phi_c ), mode='3D')

#%% Training error (training error decreases with lower order of monomials)
def l2_norm(true_state, predicted_state):
    error = true_state - predicted_state
    squaredError = np.power(error, 2)
    return np.sum(squaredError)

norms = []
for i in range(N):
    true_phi_x_prime = dPhi_Y[:,i]
    predicted_phi_x_prime = K_u(K, U[:,i]) @ Phi_X[:,i]
    norms.append(l2_norm(true_phi_x_prime, predicted_phi_x_prime))
norms = np.array(norms)

print("Mean norm on training data:", norms.mean())

#%% Define cost function
def cost(x, u):
    if len(x.shape) == 2:
        return x[0,0]**2
    return (x[0]**2) #Tried to normalize this by 10000 for overflow issues, but didn't help

#%% Discretize all controls
step_size = 0.1
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1]+step_size, step=step_size).reshape(1,-1)
#All_U = U.reshape(1,-1) # continuous case is just original domain

#%% Control
algos = algorithmsv2.algos(X, All_U, u_bounds[0], phi, psi, K, cost, epsilon=0.01, bellmanErrorType=0, weightRegularizationBool=0, u_batch_size=30)
# bellmanErrors, gradientNorms = algos.algorithm2(batch_size=64)
# algos.w = np.ones([K.shape[0],1])
algos.w = np.load('bellman-weights.npy')
print("Weights:", algos.w)

#%% Retrieve policy
def policy(x):
    pis = algos.pis(x)
    # pis = pis + ((1 - np.sum(pis)) / pis.shape[0])
    # Select action column at index sampled from policy distribution
    action = All_U[:,np.random.choice(np.arange(All_U.shape[1]), p=pis)].reshape(-1,1)
    return action

#%% Test policy

# Create the figure
fig, ax = plt.subplots()
line, = plt.plot(x, f(x, init_beta, init_c), lw=2)
point, = plt.plot(y, f(y, init_beta, init_c), 'r.', markersize=20)
ax.set_xlabel('x')
plt.ylim([-5, 5])

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

episodes = 1
steps = 1000
costs = []
for episode in range(episodes):
    starting_x = np.vstack(X[:,0]) # Maybe pick randomly?
    x = starting_x
    cost_sum = 0
    print("Initial x:", x)
    for step in range(steps):
        s.c = policy(x)[0,0]
        cost_sum += cost(x, s.c)
        y = x + s.b(x)*h + s.sigma(x)*np.sqrt(h)*np.random.randn()
        
        # line.set_ydata(f(x, s.beta, s.c))
        # point.set_xdata(y)
        # point.set_ydata(f(y, s.beta, s.c))
        
        # fig.canvas.draw_idle()

        x = y
        print("Current x:", x)
    costs.append(cost_sum)
print("Mean cost per episode:", np.mean(costs))

#%%