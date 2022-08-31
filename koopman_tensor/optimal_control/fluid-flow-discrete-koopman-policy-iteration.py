#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import care
from generalized.discrete_koopman_policy_iteration_policy import DiscreteKoopmanPolicyIterationPolicy
from scipy.integrate import solve_ivp

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% Initialize environment
state_dim = 3
action_dim = 1

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

omega = 1.0
mu = 0.1
A = -0.1
lamb = 1

dt = 0.01

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

        x_dot = mu*x - omega*y + A*x*z
        y_dot = omega*x + mu*y + A*y*z
        z_dot = -lamb * ( z - np.power(x, 2) - np.power(y, 2) )

        u = action
        if u is None:
            u = np.random.choice(all_actions, size=action_column_shape)

        return [ x_dot, y_dot + u, z_dot ]

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

#%% Define cost
Q = np.eye(state_dim)
R = 0.0001
w_r = np.array([
    [0.0],
    [0.0],
    [0.0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns are snapshots
    # x.T Q x + u.T R u
    x_ = x - w_r
    mat = np.vstack(np.diag(x_.T @ Q @ x_)) + np.power(u, 2)*R
    return mat.T

#%% Initialize important vars
state_range = 25.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

action_range = 10.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

gamma = 0.99
reg_lambda = 1.0

#%% Optimal policy
x_bar = 0
y_bar = 0
z_bar = 0
continuous_A = np.array([
    [mu + A * z_bar, -omega, A * x_bar],
    [omega, mu + A * z_bar, A * y_bar],
    [2 * lamb * x_bar, 2 * lamb * y_bar, -lamb]
])
continuous_B = np.array([
    [0],
    [1],
    [0]
])
P = care(continuous_A*np.sqrt(gamma), continuous_B*np.sqrt(gamma), Q, R)[0]
C = np.linalg.inv(R + gamma*continuous_B.T @ P @ continuous_B) @ (gamma*continuous_B.T @ P @ continuous_A)
sigma_t = reg_lambda * np.linalg.inv(R + continuous_B.T @ P @ continuous_B)

def optimal_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Construct datasets
num_episodes = 500
num_steps_per_episode = 1000
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_xs = np.zeros([num_episodes, state_dim])
for episode in range(num_episodes):
    x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 50.0], y0=x[:,0], method='RK45')
    initial_xs[episode] = soln.y[:,-1]

for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.random.choice(all_actions, size=action_column_shape)
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

#%% Compute optimal policy
policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'fluid-flow-policy-iteration.pt',
    dt=dt
)
policy.reinforce(num_training_episodes=1000, num_steps_per_episode=int(20.0 / dt))

#%% Test
test_steps = int(50.0 / dt)
def watch_agent():
    optimal_states = np.zeros([num_episodes,test_steps,state_dim])
    learned_states = np.zeros([num_episodes,test_steps,state_dim])
    optimal_actions = np.zeros([num_episodes,test_steps,action_dim])
    learned_actions = np.zeros([num_episodes,test_steps,action_dim])
    optimal_costs = np.zeros([num_episodes])
    learned_costs = np.zeros([num_episodes])

    for episode in range(num_episodes):
        state = np.random.rand(state_dim,1)*state_range*np.random.choice(np.array([-1,1]), size=(state_dim,1))
        optimal_state = state
        learned_state = state
        step = 0
        while step < test_steps:
            optimal_states[episode,step] = optimal_state[:,0]
            learned_states[episode,step] = learned_state[:,0]

            optimal_action = optimal_policy(optimal_state)
            optimal_state = f(optimal_state, optimal_action)
            optimal_costs[episode] += cost(optimal_state, optimal_action)

            with torch.no_grad():
                learned_u, _ = policy.get_action(learned_state[:,0])
            learned_action = np.array([[learned_u]])
            learned_state = f(learned_state, learned_action)
            learned_costs[episode] += cost(learned_state, learned_action)

            optimal_actions[episode,step] = optimal_action[:,0]
            learned_actions[episode,step] = learned_action[:,0]

            step += 1

    print("Norm between entire path (final episode):", utilities.l2_norm(optimal_states[-1], learned_states[-1]))
    print(f"Average cost per episode (lqr controller): {np.mean(optimal_costs)}")
    print(f"Average cost per episode (learned controller): {np.mean(learned_costs)}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('LQR Controller')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Koopman Controller')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
    for i in range(state_dim):
        axs[0].plot(optimal_states[-1,:,i], label=labels[i])
        axs[1].plot(learned_states[-1,:,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

    ax = plt.axes(projection='3d')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.plot3D(learned_states[-1,:,0], learned_states[-1,:,1], learned_states[-1,:,2], 'gray')
    plt.show()

    labels = ['optimal', 'learned']

    plt.hist(optimal_actions[-1,:,0])
    plt.hist(learned_actions[-1,:,0])
    plt.legend(labels)
    plt.show()

    plt.scatter(np.arange(optimal_actions.shape[1]), optimal_actions[-1,:,0], s=5)
    plt.scatter(np.arange(learned_actions.shape[1]), learned_actions[-1,:,0], s=5)
    plt.show()

watch_agent()

#%%