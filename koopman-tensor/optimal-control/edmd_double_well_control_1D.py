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

#%% GENERATE TRAINING DATA FOR FIXED U
class DoubleWell():
    def __init__(self, beta, c):
        self.beta = beta
        self.c = c

    def b(self, x):
        return -x**3 + 2*x + self.c
    
    def sigma(self, x):
        return np.sqrt(2/self.beta)

class EulerMaruyama(object):
    def __init__(self, h, nSteps):
        self.h = h
        self.nSteps = nSteps
    
    def integrate(self, s, x):
        y = np.zeros(self.nSteps)
        y[0] = x
        for i in range(1, self.nSteps):
            y[i] = y[i-1] + s.b(y[i-1])*self.h + s.sigma(y[i-1])*np.sqrt(self.h)*np.random.randn()
        return y

# The parametrized function to be plotted
def f(x, beta, c):
    return 1/4*x**4 - x**2 - c*x

#%% Create double-well system and integrator
init_beta = 1 # was 2
init_c = 0
h = 1e-3 # changed from 1e-2
num_steps_in_double_well = 10000
s = DoubleWell(beta=init_beta, c=init_c)
em = EulerMaruyama(h, num_steps_in_double_well)

# Starting point
x0 = 5*np.random.rand() - 2.5

#%% Generate one trajectory
y = em.integrate(s, x0)
# plt.clf()
# plt.plot(y)

#%% Generate training data
u_bounds = np.array([[-2, 2]])
x_bounds = np.array([[-1.5, 1.5]])

m = 1000#0 # number of data points
X = 5*np.random.rand(1,m) - 2.5
Y = np.zeros((1,m))
U = np.zeros((1,m))
for i in range(m):
    s.c = np.random.uniform(-2.0, 2.0)
    U[0,i] = s.c
    y = em.integrate(s, X[0,i])
    Y[0,i] = y[-1]

# plt.figure()
# plt.hist(Y, 50)

#%% Koopman Tensor
order = 6
phi = observables.monomials(order)
psi = observables.monomials(order)

#%% Build Phi and Psi matrices
Phi_X = phi(X)
Phi_Y = phi(Y)
Psi_U = psi(U)
dim_phi = Phi_X[:,0].shape[0]
dim_psi = Psi_U[:,0].shape[0]
N = X.shape[1]

print(Phi_X.shape)

#%% Build kronMatrix
kronMatrix = np.empty((dim_psi * dim_phi, N))
for i in range(N):
    kronMatrix[:,i] = np.kron(Psi_U[:,i], Phi_X[:,i])

#%% Estimate M
M = estimate_L.ols(kronMatrix.T, Phi_Y.T).T
B = estimate_L.ols(Phi_X.T, X.T)

#%% Reshape M into K tensor
K = np.empty((dim_phi, dim_phi, dim_psi))
for i in range(dim_phi):
    K[i] = M[i].reshape((dim_phi,dim_psi), order='F')

def K_u(K, u):
    psi_u = psi(u.reshape(-1,1))[:,0]
    return np.einsum('ijz,z->ij', K, psi_u)

#%% Define cost function
def cost(x, u):
    if len(x.shape) == 2:
        return x[0,0]**2
    return x[0]**2

#%% Discretize all controls
step_size = 0.1
All_U = np.arange(start=u_bounds[0,0], stop=u_bounds[0,1]+step_size, step=step_size).reshape(1,-1)
#All_U = U.reshape(1,-1) # continuous case is just original domain

#%% Learn control
algos = algorithmsv2.algos(X, All_U, u_bounds[0], phi, psi, K, cost, epsilon=0.01, bellmanErrorType=0, weightRegularizationBool=0, u_batch_size=30)
# algos.w = np.load('bellman-weights.npy')
algos.w = np.array([[-3.69297848e+00],
                    [-2.98691215e-03],
                    [ 9.07953885e-01],
                    [-2.73630256e-03],
                    [ 1.14440957e-01],
                    [ 1.56593661e-03],
                    [-3.75790976e-02]])
bellmanErrors, gradientNorms = algos.algorithm2(batch_size=64)
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
# fig, ax = plt.subplots()
# line, = plt.plot(x0, f(x0, init_beta, init_c), lw=2)
# point, = plt.plot(y, f(y, init_beta, init_c), 'r.', markersize=20)
# ax.set_xlabel('x')
# plt.ylim([-5, 5])

# axcolor = 'lightgoldenrodyellow'
# ax.margins(x=0)

# # Simulate system
# episodes = 1
# steps = 1000
# costs = []
# for episode in range(episodes):
#     starting_x = np.vstack(X[:,0]) # Maybe pick randomly?
#     x = starting_x
#     cost_sum = 0
#     print("Initial x:", x)
#     for step in range(steps):
#         s.c = policy(x)[0,0]
#         cost_sum += cost(x, s.c)
#         y = x + s.b(x)*h + s.sigma(x)*np.sqrt(h)*np.random.randn()
        
#         # line.set_ydata(f(x, s.beta, s.c))
#         # point.set_xdata(y)
#         # point.set_ydata(f(y, s.beta, s.c))
        
#         # fig.canvas.draw_idle()

#         x = y
#         if not step%250:
#             print("Current x:", x)
#     costs.append(cost_sum)
# print("Mean cost per episode:", np.mean(costs))

#%%