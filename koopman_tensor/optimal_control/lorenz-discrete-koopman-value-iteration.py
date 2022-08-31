#%% Imports
from tkinter import W
import numpy as np
import torch

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import care
from generalized.discrete_koopman_value_iteration_policy import DiscreteKoopmanValueIterationPolicy
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

PATH = './lorenz-value-model.pt'

#%% System dynamics
state_dim = 3
action_dim = 1

state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

phi_column_shape = [phi_dim, 1]

state_range = 25.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

action_range = 75.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

#%% Rest of dynamics
sigma = 10
rho = 28
beta = 8/3

dt = 0.01

x_e = np.sqrt( beta * ( rho - 1 ) )
y_e = np.sqrt( beta * ( rho - 1 ) )
z_e = rho - 1

def continuous_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - state vector
            t - timestep
        """
        x, y, z = input

        # Coordinate shifting
        # x = x + x_e
        # y = y + y_e
        # z = z + z_e

        x_dot = sigma * ( y - x )   # sigma*y - sigma*x
        y_dot = ( rho - z ) * x - y # rho*x - x*z - y
        z_dot = x * y - beta * z    # x*y - beta*z

        u = action
        if u is None:
            u = random_policy(x_dot)

        return [ x_dot + u, y_dot, z_dot ]

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[0, dt], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

x_bar = x_e
y_bar = y_e
z_bar = z_e
continuous_A = np.array([
    [-sigma, sigma, 0],
    [rho - z_bar, -1, 0],
    [y_bar, x_bar, -beta]
])
continuous_B = np.array([
    [1],
    [0],
    [0]
])
W,V = np.linalg.eig(continuous_A)
print("Eigenvalues of continuous A:\n", W)
print("Eigenvectors of continuous A:\n", V)

#%% Cost function
w_r = np.array([
    [x_e],
    [y_e],
    [z_e]
])
# w_r = np.array([
#     [-x_e],
#     [-y_e],
#     [z_e]
# ])
# w_r = np.array([
#     [0.0],
#     [0.0],
#     [0.0]
# ])
Q = np.eye(state_dim)
R = 0.001
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    # x.T @ Q @ x + u.T @ R @ u
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    return mat.T

#%% Solve riccati equation
gamma = 0.99
lamb = 1

P = care(continuous_A*np.sqrt(gamma), continuous_B*np.sqrt(gamma), Q, R)[0]
C = np.linalg.inv(R + gamma*continuous_B.T @ P @ continuous_B) @ (gamma*continuous_B.T @ P @ continuous_A)
sigma_t = lamb * np.linalg.inv(R + continuous_B.T @ P @ continuous_B)

#%% Default policy functions
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

def lqr_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Generate data
num_episodes = 500
num_steps_per_episode = int(50.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

# initial_x = np.array([[-8], [-8], [27]])
# initial_x = np.array([[-x_e], [-y_e], [z_e]])
# initial_x = np.array([[0], [1], [1.05]])
# initial_x = np.array([[0 + x_e], [1 + y_e], [1.05 + z_e]])
initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
)

for episode in range(num_episodes):
    # x = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
    x = np.vstack(initial_states[:, episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = random_policy(x)
        U[:,(episode*num_steps_per_episode)+step] = u[:,0]
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Run Algorithm
policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    lamb,
    tensor,
    all_actions,
    cost,
    'lorenz-value-iteration.pt',
    dt
)
policy.train(
    training_epochs=500,
    batch_size=2**9,
    batch_scale=3,
    epsilon=1e-2,
    gamma_increment_amount=0.02
)

#%% Test policy in environment
def watch_agent(num_episodes, step_limit):
    lqr_states = np.zeros([num_episodes,state_dim,step_limit])
    lqr_actions = np.zeros([num_episodes,action_dim,step_limit])
    lqr_costs = torch.zeros([num_episodes])

    koopman_states = np.zeros([num_episodes,state_dim,step_limit])
    koopman_actions = np.zeros([num_episodes,action_dim,step_limit])
    koopman_costs = torch.zeros([num_episodes])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    )

    for episode in range(num_episodes):
        # state = np.vstack(initial_xs[episode]) # to start with different initial condition, but same policy
        # state = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
        # state = np.array([[-x_e],[-y_e],[z_e]])
        state = np.vstack(initial_states[:, episode])

        lqr_state = state
        koopman_state = state

        lqr_cumulative_cost = 0
        koopman_cumulative_cost = 0

        for step in range(step_limit):
            lqr_states[episode,:,step] = lqr_state[:,0]
            koopman_states[episode,:,step] = koopman_state[:,0]

            lqr_action = lqr_policy(state)
            # if lqr_action[0,0] > action_range:
            #     lqr_action = np.array([[action_range]])
            # elif lqr_action[0,0] < -action_range:
            #     lqr_action = np.array([[-action_range]])
            lqr_actions[episode,:,step] = lqr_action

            koopman_action = policy.get_action(state)
            koopman_actions[episode,:,step] = koopman_action

            lqr_cumulative_cost += cost(lqr_state, lqr_action)[0,0]
            koopman_cumulative_cost += cost(koopman_state, lqr_action)[0,0]

            lqr_state = f(lqr_state, lqr_action)
            koopman_state = f(koopman_state, koopman_action)

        lqr_costs[episode] = lqr_cumulative_cost
        koopman_costs[episode] = koopman_cumulative_cost

    print(f"Mean cost per episode over {num_episodes} episode(s) (LQR controller): {torch.mean(lqr_costs)}")
    print(f"Mean cost per episode over {num_episodes} episode(s) (Koopman controller): {torch.mean(koopman_costs)}\n")

    print(f"Initial state of final episode (LQR controller): {lqr_states[-1,:,0]}")
    print(f"Final state of final episode (LQR controller): {lqr_states[-1,:,-1]}\n")

    print(f"Initial state of final episode (Koopman controller): {koopman_states[-1,:,0]}")
    print(f"Final state of final episode (Koopman controller): {koopman_states[-1,:,-1]}\n")

    print(f"Reference state: {w_r[:,0]}\n")

    print(f"Difference between final state of final episode and reference state (LQR controller): {np.abs(lqr_states[-1,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of final episode and reference state (LQR controller): {utilities.l2_norm(lqr_states[-1,:,-1], w_r[:,0])}\n")

    print(f"Difference between final state of final episode and reference state (Koopman controller): {np.abs(koopman_states[-1,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of final episode and reference state (Koopman controller): {utilities.l2_norm(koopman_states[-1,:,-1], w_r[:,0])}")

    ax = plt.axes(projection='3d')
    ax.set_xlim(-20.0, 20.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_zlim(0.0, 50.0)
    ax.plot3D(lqr_states[-1,:,0], lqr_states[-1,:,1], lqr_states[-1,:,2], 'gray')
    plt.title("LQR Controller in Environment (3D)")
    plt.show()

    ax = plt.axes(projection='3d')
    ax.set_xlim(-20.0, 20.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_zlim(0.0, 50.0)
    ax.plot3D(koopman_states[-1,:,0], koopman_states[-1,:,1], koopman_states[-1,:,2], 'gray')
    plt.title("Koopman Controller in Environment (3D)")
    plt.show()

    labels = ['LQR controller', 'Koopman controller']

    plt.hist(lqr_actions[-1,:,0])
    plt.hist(koopman_actions[-1,:,0])
    plt.legend(labels)
    plt.show()

    plt.scatter(np.arange(lqr_actions.shape[1]), lqr_actions[-1,:,0], s=5)
    plt.scatter(np.arange(koopman_actions.shape[1]), koopman_actions[-1,:,0], s=5)
    plt.legend(labels)
    plt.show()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=int(50.0 / dt))

#%%
