#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import dare
from generalized.discrete_koopman_policy_iteration_policy import DiscreteKoopmanPolicyIterationPolicy
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables
import utilities

#%% Initialize environment
state_dim = 4
action_dim = 1

A = np.zeros([state_dim,state_dim])
max_abs_real_eigen_val = 1.0
while max_abs_real_eigen_val >= 1.0 or max_abs_real_eigen_val <= 0.7:
    Z = np.random.rand(*A.shape)
    A = Z.T @ Z
    W,V = np.linalg.eig(A)
    max_abs_real_eigen_val = np.max(np.abs(np.real(W)))
# A = np.array([
#     [1.0, 0.02,  0.0,        0.0 ],
#     [0.0, 1.0,  -0.01434146, 0.0 ],
#     [0.0, 0.0,   1.0,        0.02],
#     [0.0, 0.0,   0.3155122,  1.0 ]
# ])
# A = np.array([
#     [0.35782475, 0.17592175, 0.3477801, 0.22458724],
#     [0.17592175, 0.10036655, 0.11508892, 0.12756766],
#     [0.3477801,  0.11508892, 0.84861714, 0.18164662],
#     [0.22458724, 0.12756766, 0.18164662, 0.19736555]
# ])
W,V = np.linalg.eig(A)
max_real_eigen_val = np.max(np.abs(np.real(W)))
print("A:", A)
print("A's max absolute real eigenvalue:", max_abs_real_eigen_val)
B = np.ones([state_dim,action_dim])

def f(x, u):
    return A @ x + B @ u

#%% Define cost
Q_ = np.eye(state_dim)
R = 1
w_r = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0]
])
# w_r = np.array([
#     [2.0],
#     [2.0],
#     [2.0],
#     [2.0]
# ])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns are snapshots
    # x.T Q x + u.T R u
    x_ = x - w_r
    mat = np.vstack(np.diag(x_.T @ Q_ @ x_)) + np.power(u, 2)*R
    return mat.T

#%% Initialize important vars
state_range = 25.0
minimum_state = np.ones([state_dim,1]) * -state_range
maximum_state = np.ones([state_dim,1]) * state_range

action_range = 5.0
minimum_action = np.ones([action_dim,1]) * -action_range
maximum_action = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

step_size = 0.1
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

gamma = 0.99
lamb = 1.0

#%% Optimal policy
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q_, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)

def optimal_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Construct datasets
num_episodes = 100
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(minimum_state, maximum_state, [state_dim,N])
U = np.random.uniform(minimum_action, maximum_action, [action_dim,N])
Y = f(X, U)

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
#! For some reason, this is not working. It keeps computing cost is 800,000+ when it should be something like 4,000
policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    lamb,
    tensor,
    minimum_state,
    maximum_state,
    all_actions,
    cost,
    'lqr-policy-iteration.pt'
)
policy.reinforce(num_training_episodes=500, num_steps_per_episode=200)

#%% Test
test_steps = 200
def watch_agent():
    optimal_states = np.zeros([num_episodes,test_steps,state_dim])
    learned_states = np.zeros([num_episodes,test_steps,state_dim])
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

            step += 1

    print("Norm between entire path (final episode):", utilities.l2_norm(optimal_states[-1], learned_states[-1]))
    print(f"Average cost per episode (optimal controller): {np.mean(optimal_costs)}")
    print(f"Average cost per episode (learned controller): {np.mean(learned_costs)}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('True dynamics')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Learned dynamics')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = np.array(['x_0', 'x_1', 'x_2', 'x_3'])
    for i in range(A.shape[1]):
        axs[0].plot(optimal_states[-1,:,i], label=labels[i])
        axs[1].plot(learned_states[-1,:,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

watch_agent()

#%%