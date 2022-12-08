# Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.integrate import solve_ivp

import sys
sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor

# Variables
state_dim = 3
action_dim = 1

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

state_order = 2
action_order = 2

# Default basic policies
def zero_policy(x=None):
    return np.zeros(action_column_shape)

# Dynamics
omega = 1.0
mu = 0.1
A = -0.1
lamb = 1

dt = 0.01

# Define the fluid flow system
def continuous_f(action=None):
    """
        True, continuous dynamics of the system.

        INPUTS:
            action - Action vector. If left as None, then random policy is used.
    """

    def f_u(t, input):
        """
            INPUTS:
                input - State vector.
                t - Timestep.
        """

        x, y, z = input

        x_dot = mu*x - omega*y + A*x*z
        y_dot = omega*x + mu*y + A*y*z
        z_dot = -lamb * ( z - np.power(x, 2) - np.power(y, 2) )

        u = action
        if u is None:
            u = zero_policy()

        return [ x_dot, y_dot + u, z_dot ]

    return f_u

def f(state, action):
    """
        True, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.

        INPUTS:
            state - State column vector.
            action - Action column vector.

        OUTPUTS:
            State column vector pushed forward in time.
    """

    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[0, dt], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

# Define the grid spacing
# dx = 0.1
# dy = 0.1
dx = 0.5
dy = 0.5

# Create the x and y arrays
x = np.arange(-1, 1, dx)
y = np.arange(-1, 1, dy)
X, Y = np.meshgrid(x, y)

# Set the initial values and time step
dt = 0.01
num_timesteps = 10.0
step_cnt = int(num_timesteps / dt)

# Set initial state values
# initial_state = np.array([
#     [0.2],
#     [0.2],
#     [1.0]
# ])
# initial_states = np.empty((state_dim, X.shape[0], Y.shape[0]))
# for i in range(X.shape[0]):
#     for j in range(Y.shape[0]):
#         initial_states[:, i, j] = np.array([X[i,j], Y[i,j], 0])

# Generate path
# states = odeint(
#     func=continuous_f(None),
#     y0=initial_state[:,0],
#     t=np.arange(0, num_timesteps, dt),
#     tfirst=True
# ).T
# states = np.empty((state_dim, step_cnt, X.shape[0], Y.shape[0]))
# for i in range(initial_states.shape[0]):
#     for j in range(initial_states.shape[1]):
#         states[:, 0, i, j] = initial_states[:, i, j]
# for step in range(1, step_cnt):
#     for i in range(X.shape[0]):
#         for j in range(Y.shape[0]):
#             state = np.vstack(states[:, step-1, i, j])
#             action = np.array([[0]])
#             states[:, step, i, j] = f(state, action)[:,0]

# Plot the path
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.sca(ax)
# for i in range(X.shape[0]):
#     for j in range(Y.shape[0]):
#         plt.plot(
#             states[0,:,i,j],
#             states[1,:,i,j],
#             states[2,:,i,j]
#         )
# plt.title("Path in System")
# plt.show()

# Estimate Koopman tensor
# inputs = states.reshape(state_dim, step_cnt*X.shape[0]*Y.shape[0])
# outputs = np.roll(inputs, -1, axis=1)[:,:-1]
# inputs = inputs[:,:-1]
# actions = np.zeros((action_dim,inputs.shape[1]))
def identity(input):
    return input
# tensor = KoopmanTensor(
#     inputs,
#     outputs,
#     actions,
#     phi=identity, # observables.monomials(state_order),
#     psi=observables.monomials(action_order),
#     regressor='ols'
# )
# with open('tensor.pickle', 'wb') as f:
#     pickle.dump(tensor, f)
with open('tensor.pickle', 'rb') as f:
    tensor = pickle.load(f)

# Reshape states so as to stack vector field into a single vector for each timestamp
# states = states.reshape((state_dim, step_cnt*X.shape[0]*Y.shape[0])).T

# Compute the singular value decomposition of the system's trajectory
# U, S, V = np.linalg.svd(states)
# print(tensor.K_(np.array([[0]])))
# U, S, V = np.linalg.svd(tensor.K_(np.array([[0]])))
# print(U.shape)
# print(V.shape)

# Evaluate the POD modes at each point in the grid
# U1 = U[0, 0] * X + U[1, 0] * Y # evaluate the first POD mode at each point
# U2 = U[0, 1] * X + U[1, 1] * Y # evaluate the second POD mode at each point
# U3 = U[0, 2] * X + U[1, 2] * Y # evaluate the third POD mode at each point

# Plot the POD modes using matplotlib
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.sca(ax)
# plt.contourf(X, Y, U1) # plot the first POD mode
# plt.contourf(X, Y, U2) # plot the second POD mode
# plt.contourf(X, Y, U3) # plot the third POD mode
# plt.show()
# plt.imshow(tensor.K_(np.array([[0]])), cmap="gray") # plot the original matrix as a heat map
# plt.bar(range(len(S)), S) # plot the singular values as a bar chart
# for i in range(len(S)):
#     plt.arrow(i, 0, V[i, 0], V[i, 1], head_width=0.2, head_length=0.2) # plot the corresponding singular vectors as arrows
# plt.show()
# ax = fig.add_subplot(121)
# mode1 = U[:,0].reshape(X.shape[0], Y.shape[0], state_dim)
# mode2 = U[:,1].reshape(X.shape[0], Y.shape[0], state_dim)
# mode3 = U[:,2].reshape(X.shape[0], Y.shape[0], state_dim)
# for i in range(X.shape[0]):
#     for j in range(Y.shape[0]):
#         for dim in range(state_dim):
#             mode1[i, j, dim] = int(np.abs(mode1[i, j, dim]*100))
#             mode2[i, j, dim] = int(np.abs(mode2[i, j, dim]*100))
#             mode3[i, j, dim] = int(np.abs(mode3[i, j, dim]*100))
# ax = fig.add_subplot(131)
# plt.sca(ax)
# plt.imshow(mode1)
# ax = fig.add_subplot(132)
# plt.sca(ax)
# plt.imshow(mode2)
# ax = fig.add_subplot(133)
# plt.sca(ax)
# plt.imshow(mode3)
# plt.show()
# for i in range(3):
#     plt.plot(U[:,i])
#     plt.imshow(U[:,i], cmap='RdBu')
# plt.contourf(X, Y, U @ np.diag(S), cmap='RdBu')

# ax = fig.add_subplot(122, projection='3d')
# plt.sca(ax)
# plt.plot(U[:,0], U[:,1], U[:,2], lw=0.5)
# plt.xlabel("Mode 1")
# plt.ylabel("Mode 2")
# ax.set_zlabel("Mode 3")
# plt.title("POD Modes of the Fluid Flow")

# plt.tight_layout()
# plt.show()