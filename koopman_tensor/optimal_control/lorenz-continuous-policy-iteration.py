# import gym
from importlib.metadata import requires
import numpy as np
import scipy as sp
import time
import torch

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import lqr, care
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../')
import observables
import utilities

PATH = './lorenz-continuous-policy.pt'

#%% System dynamics
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

# state_range = 25.0
# state_minimums = np.ones([state_dim,1]) * -state_range
# state_maximums = np.ones([state_dim,1]) * state_range
state_minimums = np.array([
    [-20.0],
    [-50.0],
    [0.0]
])
state_maximums = np.array([
    [20.0],
    [50.0],
    [50.0]
])

action_range = 75.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

#%% Rest of dynamics
sigma = 10
rho = 28
beta = 8/3

dt = 0.01
t_span = np.arange(0, dt, dt/10)

x_e = np.sqrt( beta * ( rho - 1 ) )
y_e = np.sqrt( beta * ( rho - 1 ) )
z_e = rho - 1

gamma = 0.99
reg_lambda = 1.0

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

        # x = x - x_e
        # y = y - y_e
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

    soln = solve_ivp(fun=continuous_f(u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

#%% Reward function
w_r = np.array([
    [x_e],
    [y_e],
    [z_e]
])
Q_ = np.eye(state_dim)
R = 0.0001
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat.T

#%% Default policy functions
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

# LQR Policy
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

P = care(continuous_A*np.sqrt(gamma), continuous_B*np.sqrt(gamma), Q_, R)[0]
C = np.linalg.inv(R + gamma*continuous_B.T @ P @ continuous_B) @ (gamma*continuous_B.T @ P @ continuous_A)
sigma_t = reg_lambda * np.linalg.inv(R + continuous_B.T @ P @ continuous_B)

def lqr_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Generate data
num_episodes = 500
num_steps_per_episode = int(20.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

# initial_x = np.array([[0 - x_e], [1 - y_e], [1.05 + z_e]])
initial_xs = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
)

for episode in range(num_episodes):
    # x = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
    x = np.vstack(initial_xs[:,episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.random.choice(all_actions, size=action_column_shape)
        U[:,(episode*num_steps_per_episode)+step] = u[:,0]
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

#%% Estimate Koopman operator
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%% Estimate Q function for current policy
w_hat_batch_size = 2**8 # 2**12
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        # pi_response = policy_net.predict(x_batch.T).T # (all_actions.shape[0], w_hat_batch_size)
        pi_response = np.zeros([all_actions.shape[0],w_hat_batch_size])
        for state_index, state in enumerate(x_batch.T):
            action_distribution = policy_net.get_action_distribution(state.reshape(-1,1))
            pi_response[:, state_index] = action_distribution.log_prob(torch.tensor(all_actions))

    phi_x_prime_batch = tensor.K_(np.array([all_actions])) @ phi_x_batch # (all_actions.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response) # (all_actions.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = np.einsum('uw,uw->wu', -cost(x_batch, np.array([all_actions])), pi_response) # (w_hat_batch_size, all_actions.shape[0])
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    )

    return w_hat

def Q(x, u, w_hat_t):
    return (-cost(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u))[0,0]

#%% Policy function as PyTorch model
class PolicyNetwork():
    def __init__(self, input_dim, lr=0.003):
        self.alpha = torch.zeros([1,input_dim], requires_grad=True)
        self.beta = torch.zeros([1,input_dim], requires_grad=True)
        self.optimizer = torch.optim.Adam([self.alpha, self.beta], lr)

    def update(self, returns, log_probs):
        """
            Update the weights of the policy network given the training samples
            @param returns: return (cumulative rewards) for each step in an episode
            @param log_probs: log probability for each step
        """
        policy_gradient = []
        for i, (log_prob, Gt) in enumerate(zip(log_probs, returns)):
            policy_gradient.append(-log_prob * gamma**((len(returns)-i) * dt) * Gt)

        loss = torch.stack(policy_gradient).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action_distribution(self, s):
        phi_s = torch.Tensor(s) # torch.Tensor(tensor.phi(s))
        mu = (self.alpha @ phi_s)[0,0]
        sigma = torch.exp((self.beta @ phi_s)[0,0])
        return torch.distributions.normal.Normal(mu, sigma, validate_args=False)

    def sample_action(self, s):
        """
            Estimate the policy and sample an action, compute its log probability
            @param s: input state (column vector)
            @return: the selected action and log probability
        """
        action_distribution = self.get_action_distribution(s)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action, log_prob

def reinforce(estimator, n_episode, gamma=1.0):
    """
        REINFORCE algorithm
        @param estimator: policy network
        @param n_episode: number of episodes
        @param gamma: the discount factor
    """
    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    )

    w_hat = np.zeros([phi_dim,1])
    num_steps_per_trajectory = int(20.0 / dt)
    for episode in range(n_episode):
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        state = np.vstack(initial_states[:,episode])

        for step in range(num_steps_per_trajectory):
            states.append(state)

            u, log_prob = estimator.sample_action(state)
            action = np.array([[u]])
            actions.append(u)
            log_probs.append(log_prob)

            curr_reward = -cost(state, action)[0,0]
            rewards.append(curr_reward)

            total_reward_episode[episode] += gamma**(step * dt) * curr_reward 

            state = f(state, action)
        
        returns_arr = torch.zeros([len(rewards)])
        # Gt = 0
        for i in range(len(rewards)-1, -1, -1):
            # Gt = rewards[i] + (gamma * Gt)
            # returns[i] = Gt
            
            Q_val = Q(
                np.vstack(states[i]),
                np.array([[actions[i]]]),
                w_hat
            )
            returns_arr[i] = Q_val

        returns = (returns_arr - returns_arr.mean()) / (returns_arr.std() + torch.finfo(torch.float64).eps)

        estimator.update(returns, log_probs)

        if (episode+1) % 250 == 0:
            print(f"Episode: {episode+1}, avg discounted total reward: {np.mean(total_reward_episode[episode-249:episode])}")
            torch.save(estimator, PATH)

        w_hat = w_hat_t()

#%%
# n_state = env.observation_space.shape[0]
num_actions = all_actions.shape[0]

policy_net = PolicyNetwork(state_dim) # phi_dim

# policy_net = torch.load(PATH)

#%% Generate new initial xs for learning control
num_episodes = 1000
# num_episodes = 0

#%% Run REINFORCE
total_reward_episode = [0] * num_episodes
reinforce(policy_net, num_episodes, gamma)

# import matplotlib.pyplot as plt
# plt.plot(total_reward_episode)
# plt.title('Episode reward over time')
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()

#%% Test policy in environment
def watch_agent(num_episodes, test_steps):
    lqr_states = np.zeros([num_episodes,state_dim,test_steps])
    lqr_actions = np.zeros([num_episodes,action_dim,test_steps])
    lqr_costs = np.zeros([num_episodes])

    koopman_states = np.zeros([num_episodes,state_dim,test_steps])
    koopman_actions = np.zeros([num_episodes,action_dim,test_steps])
    koopman_costs = np.zeros([num_episodes])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    )

    for episode in range(num_episodes):
        # state = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
        state = np.vstack(initial_states[:, episode])

        lqr_state = state
        koopman_state = state

        for step in range(test_steps):
            # Append state to list of states
            lqr_states[episode,:,step] = lqr_state[:,0]
            koopman_states[episode,:,step] = koopman_state[:,0]

            # LQR action
            lqr_action = lqr_policy(lqr_state)
            lqr_actions[episode,:,step] = lqr_action[:,0]

            # Koopman action 
            with torch.no_grad():
                koopman_action, _ = policy_net.sample_action(state)
            koopman_action = np.array([[koopman_action]])
            koopman_actions[episode,:,step] = koopman_action

            # Add cost to accumulators
            lqr_costs[episode] += cost(lqr_state, lqr_action)[0,0]
            koopman_costs[episode] += cost(koopman_state, koopman_action)[0,0]

            # Update states
            lqr_state = f(lqr_state, lqr_action)
            koopman_state = f(koopman_state, koopman_action)

    print(f"Mean cost per episode over {num_episodes} episode(s) (LQR controller): {np.mean(lqr_costs)}")
    print(f"Mean cost per episode over {num_episodes} episode(s) (Koopman controller): {np.mean(koopman_costs)}\n")

    print(f"Initial state of final episode (LQR controller): {lqr_states[-1,:,0]}")
    print(f"Final state of final episode (LQR controller): {lqr_states[-1,:,-1]}\n")

    print(f"Initial state of final episode (Koopman controller): {koopman_states[-1,:,0]}")
    print(f"Final state of final episode (Koopman controller): {koopman_states[-1,:,-1]}\n")

    print(f"Reference state: {w_r[:,0]}\n")

    print(f"Difference between final state of final episode and reference state (LQR controller): {np.abs(lqr_states[-1,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of final episode and reference state (LQR controller): {utilities.l2_norm(lqr_states[-1,:,-1], w_r[:,0])}\n")

    print(f"Difference between final state of final episode and reference state (Koopman controller): {np.abs(koopman_states[-1,:,-1] - w_r[:,0])}")
    print(f"Norm between final state of final episode and reference state (Koopman controller): {utilities.l2_norm(koopman_states[-1,:,-1], w_r[:,0])}")

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('LQR Controller')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Koopman Controller')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        axs[0].plot(lqr_states[-1,i], label=labels[i])
        axs[1].plot(koopman_states[-1,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

    ax = plt.axes(projection='3d')
    ax.set_xlim(-20.0, 20.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_zlim(0.0, 50.0)
    ax.plot3D(lqr_states[-1,0], lqr_states[-1,1], lqr_states[-1,2], 'gray')
    plt.title("LQR Controller in Environment (3D)")
    plt.show()

    ax = plt.axes(projection='3d')
    ax.set_xlim(-20.0, 20.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_zlim(0.0, 50.0)
    ax.plot3D(koopman_states[-1,0], koopman_states[-1,1], koopman_states[-1,2], 'gray')
    plt.title("Koopman Controller in Environment (3D)")
    plt.show()

    labels = ['LQR controller', 'Koopman controller']

    plt.hist(lqr_actions[-1,0])
    plt.hist(koopman_actions[-1,0])
    plt.legend(labels)
    plt.show()

    plt.scatter(np.arange(lqr_actions.shape[2]), lqr_actions[-1,0], s=5)
    plt.scatter(np.arange(koopman_actions.shape[2]), koopman_actions[-1,0], s=5)
    plt.legend(labels)
    plt.show()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, test_steps=int(50.0 / dt))

#%%