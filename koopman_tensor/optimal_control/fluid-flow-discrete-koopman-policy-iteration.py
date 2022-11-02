#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import care
from generalized.discrete_koopman_actor_critic_policy import DiscreteKoopmanActorCriticPolicy
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
R = 0.001
# R = np.eye(action_dim) * R
w_r = np.array([
    [0.0],
    [0.0],
    [0.0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns are snapshots

    _x = x - w_r

    # return _x.T @ Q @ _x + u.T @ R @ u

    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    # mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.vstack(np.diag(u.T @ R @ u))
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
all_actions = np.array([all_actions])

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

def lqr_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Construct datasets
num_episodes = 500
num_steps_per_episode = int(10.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_xs = np.zeros([num_episodes, state_dim])
for episode in range(num_episodes):
    x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 30.0], y0=x[:,0], method='RK45')
    initial_xs[episode] = soln.y[:,-1]

for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.random.choice(all_actions[0], size=action_column_shape)
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
policy = DiscreteKoopmanActorCriticPolicy(
    f,
    gamma,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'fluid-flow-policy-iteration.pt',
    dt=dt
)
policy.train(
    num_training_episodes=1000,
    num_steps_per_episode=int(20.0 / dt)
)

#%% Test agent
def watch_agent(num_episodes, test_steps):
    # specifiedEpisode = -1
    specifiedEpisode = 42

    lqr_states = np.zeros([num_episodes,state_dim,test_steps])
    lqr_actions = np.zeros([num_episodes,action_dim,test_steps])
    lqr_costs = np.zeros([num_episodes])

    koopman_states = np.zeros([num_episodes,state_dim,test_steps])
    koopman_actions = np.zeros([num_episodes,action_dim,test_steps])
    koopman_costs = np.zeros([num_episodes])

    initial_xs = np.zeros([num_episodes, state_dim])
    for episode in range(num_episodes):
        x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1, 1], size=state_column_shape)
        u = np.array([[0]])
        soln = solve_ivp(fun=continuous_f(u), t_span=[0, 30.0], y0=x[:,0], method='RK45')
        initial_xs[episode] = soln.y[:,-1]

    for episode in range(num_episodes):
        # Get initial state from precomputed states
        state = np.vstack(initial_xs[episode])
        
        # Split state for LQR and actor-critic controllers
        lqr_state = state
        koopman_state = state

        for step in range(test_steps):
            # Store states
            lqr_states[episode,:,step] = lqr_state[:,0]
            koopman_states[episode,:,step] = koopman_state[:,0]

            # Get action from LQR policy
            lqr_action = lqr_policy(lqr_state)
            # if lqr_action[0,0] > action_range:
            #     lqr_action = np.array([[action_range]])
            # elif lqr_action[0,0] < -action_range:
            #     lqr_action = np.array([[-action_range]])
            lqr_actions[episode,:,step] = lqr_action[:,0]

            # Get action from actor-critic policy
            with torch.no_grad():
                koopman_action, _ = policy.get_action(koopman_state)
            koopman_actions[episode,:,step] = koopman_action[:,0]

            # Compute costs
            lqr_costs[episode] += cost(lqr_state, lqr_action)[0,0]
            koopman_costs[episode] += cost(koopman_state, koopman_action)[0,0]

            # Compute next states
            lqr_state = f(lqr_state, lqr_action)
            koopman_state = f(koopman_state, koopman_action)

    print(f"Mean cost per episode over {num_episodes} episode(s) (LQR controller): {np.mean(lqr_costs)}")
    print(f"Mean cost per episode over {num_episodes} episode(s) (Koopman controller): {np.mean(koopman_costs)}\n")

    print(f"Initial state of episode #{specifiedEpisode} (LQR controller): {lqr_states[specifiedEpisode,:,0]}")
    print(f"Final state of episode #{specifiedEpisode} (LQR controller): {lqr_states[specifiedEpisode,:,-1]}\n")

    print(f"Initial state of episode #{specifiedEpisode} (Koopman controller): {koopman_states[specifiedEpisode,:,0]}")
    print(f"Final state of episode #{specifiedEpisode} (Koopman controller): {koopman_states[specifiedEpisode,:,-1]}\n")

    print(f"Reference state: {w_r[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (LQR controller): {np.abs(lqr_states[specifiedEpisode,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (LQR controller): {utilities.l2_norm(lqr_states[specifiedEpisode,:,-1], w_r[:,0])}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {np.abs(koopman_states[specifiedEpisode,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state (Koopman controller): {utilities.l2_norm(koopman_states[specifiedEpisode,:,-1], w_r[:,0])}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('LQR Controller')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Koopman Controller')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        axs[0].plot(lqr_states[specifiedEpisode,i], label=labels[i])
        axs[1].plot(koopman_states[specifiedEpisode,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.plot(
        lqr_states[specifiedEpisode,0],
        lqr_states[specifiedEpisode,1],
        lqr_states[specifiedEpisode,2],
        'gray'
    )
    ax.set_title("LQR Controller in Environment (3D)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.plot(
        koopman_states[specifiedEpisode,0],
        koopman_states[specifiedEpisode,1],
        koopman_states[specifiedEpisode,2],
        'gray'
    )
    ax.set_title("Koopman Controller in Environment (3D)")
    plt.show()

    labels = ['LQR controller', 'Koopman controller']

    plt.hist(lqr_actions[specifiedEpisode,0])
    plt.hist(koopman_actions[specifiedEpisode,0])
    plt.legend(labels)
    plt.title(f"Histogram Of Actions (In Episode #{specifiedEpisode})")
    plt.show()

    plt.scatter(np.arange(lqr_actions.shape[2]), lqr_actions[-1,0], s=5)
    plt.scatter(np.arange(koopman_actions.shape[2]), koopman_actions[-1,0], s=5)
    plt.legend(labels)
    plt.title(f"Scatter Plot of Actions Per Step (In Episode #{specifiedEpisode})")
    plt.show()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, test_steps=int(30.0 / dt))