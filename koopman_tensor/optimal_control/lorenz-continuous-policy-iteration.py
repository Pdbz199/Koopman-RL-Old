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

action_range = np.array([-75, 75])
step_size = 1.0
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
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

        x = x - x_e
        y = y - y_e
        z = z + z_e

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
    [0],
    [0],
    [0] # 1
])
Q_ = np.eye(state_dim)
R = 0.0001
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat.T
def reward(x, u):
    return -cost(x, u)

#%% Default policy functions
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

#%% Generate data
num_episodes = 500
num_steps_per_episode = 1000
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

initial_x = np.array([[0 - x_e], [1 - y_e], [1.05 + z_e]])

for episode in range(num_episodes):
    x = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
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
w_hat_batch_size = 2**4 # 2**12
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

    reward_batch_prob = np.einsum('uw,uw->wu', reward(x_batch, np.array([all_actions])), pi_response) # (w_hat_batch_size, all_actions.shape[0])
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    )

    return w_hat

def Q(x, u, w_hat_t):
    return (reward(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u))[0,0]

#%% Policy function as PyTorch model
class PolicyNetwork():
    def __init__(self, phi_dim, lr=0.003):
        self.alpha = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.alpha], lr)

        self.w_hat = np.zeros([phi_dim,1])

    def update(self, returns, log_probs):
        """
            Update the weights of the policy network given the training samples
            @param returns: return (cumulative rewards) for each step in an episode
            @param log_probs: log probability for each step
        """
        policy_gradient = []
        for i, (log_prob, Gt) in enumerate(zip(log_probs, returns)):
            policy_gradient.append(-log_prob * gamma**(len(returns)-i) * Gt)

        loss = torch.stack(policy_gradient).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.w_hat = w_hat_t()

    def get_action_distribution(self, s):
        mu = (self.w_hat.T @ tensor.phi(s))[0,0]
        sigma = torch.exp(self.alpha)
        return torch.distributions.normal.Normal(mu, sigma, [])

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
    num_steps_per_trajectory = 500
    for episode in range(n_episode):
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        state = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))

        step = 0
        while len(rewards) < num_steps_per_trajectory:
            u, log_prob = estimator.sample_action(state)
            action = np.array([[u]])
            next_state = tensor.f(state, action) # tensor.f(state, action) for speed?
            curr_reward = reward(state, action)[0,0]

            states.append(state)
            actions.append(u)

            total_reward_episode[episode] += gamma**step * curr_reward 
            log_probs.append(log_prob)
            rewards.append(curr_reward)

            if len(rewards) == num_steps_per_trajectory:
                returns = torch.zeros([len(rewards)])
                # Gt = 0
                for i in range(len(rewards)-1, -1, -1):
                    # Gt = rewards[i] + (gamma * Gt)
                    # returns[i] = Gt
                    
                    Q_val = Q(
                        np.vstack(states[i]),
                        np.array([[actions[i]]]),
                        estimator.w_hat
                    )
                    returns[i] = Q_val

                returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

                estimator.update(returns, log_probs)
                if (episode+1) % 250 == 0:
                    print(f"Episode: {episode+1}, discounted total reward: {total_reward_episode[episode]}")
                    torch.save(estimator, PATH)

                break

            print(f"step: {step}")

            step += 1
            state = next_state

#%%
# n_state = env.observation_space.shape[0]
num_actions = all_actions.shape[0]

policy_net = PolicyNetwork(phi_dim)

# policy_net = torch.load(PATH)

#%% Generate new initial xs for learning control
num_episodes = 2000
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
num_episodes = 100

step_limit = 400 # 10000
def watch_agent():
    states = np.zeros([num_episodes,state_dim,step_limit])
    actions = np.zeros([num_episodes,action_dim,step_limit])
    costs = torch.zeros([num_episodes])
    for episode in range(num_episodes):
        state = initial_x + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))

        states[episode,:,0] = state[:,0]
        cumulative_cost = 0
        step = 0
        while step < step_limit:
            with torch.no_grad():
                action, _ = policy_net.sample_action(state)
                action = np.array([[action]])
            #     action = lqr_policy(state)
            # if action[0,0] > action_range[1]:
            #     action = np.array([[action_range[1]]])
            # if action[0,0] < action_range[0]:
            #     action = np.array([[action_range[0]]])
            state = tensor.f(state, action)
            states[episode,:,step] = state[:,0]
            actions[episode,:,step] = action
            cumulative_cost += cost(state, action)[0,0]
            step += 1
            if step == step_limit:
                costs[episode] = cumulative_cost
                # print(f"Total cost for episode {episode}:", cumulative_cost)
    print(f"Mean cost per episode over {num_episodes} episodes:", torch.mean(costs))
    print("Initial state of final episode:", states[-1,:,0])
    print("Final state of final episode:", states[-1,:,-1])
    print("Difference between final state of final episode and reference state:", np.abs(states[-1,:,-1] - w_r[:,0]))

    ax = plt.axes(projection='3d')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.plot3D(states[-1,0], states[-1,1], states[-1,2], 'gray')
    plt.show()

    plt.scatter(np.arange(actions.shape[2]), actions[-1,0], s=10)
    plt.show()
print("Testing learned policy...")
watch_agent()

#%%