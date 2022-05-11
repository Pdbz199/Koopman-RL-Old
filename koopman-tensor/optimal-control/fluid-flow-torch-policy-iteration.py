# import gym
import numpy as np
import scipy as sp
import torch
import torch.nn as nn

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from scipy.integrate import solve_ivp

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../')
import observables

PATH = './fluid-flow-policy.pt'

#%% System dynamics
x_dim = 3
x_column_shape = [x_dim, 1]
u_dim = 1
u_column_shape = [u_dim, 1]
state_order = 2
action_order = 2
state_M_plus_N_minus_ones = np.arange( (x_dim-1), state_order + (x_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( state_M_plus_N_minus_ones, np.ones_like(state_M_plus_N_minus_ones) * ( x_dim-1 ) ) ) )
action_M_plus_N_minus_ones = np.arange( (u_dim-1), action_order + (u_dim-1) + 1 )
phi_dim = int( np.sum( sp.special.comb( action_M_plus_N_minus_ones, np.ones_like(action_M_plus_N_minus_ones) * ( u_dim-1 ) ) ) )

u_range = torch.Tensor([-1, 1])
all_us = torch.arange(u_range[0], u_range[1], 0.01) #* if too compute heavy, 0.05
omega = 1.0
mu = 0.1
A = -0.1
lamb = 1
t_span = np.arange(0, 0.001, 0.0001)

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
            u = np.random.choice(all_us, size=u_column_shape)

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

    soln = solve_ivp(fun=continuous_f(u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

#%% Generate data
num_episodes = 500
num_steps_per_episode = 1000
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([x_dim,N])
Y = np.zeros([x_dim,N])
U = np.zeros([u_dim,N])

initial_xs = np.zeros([num_episodes, x_dim])
for episode in range(num_episodes):
    x = np.random.random(x_column_shape) * 0.5 * np.random.choice([-1,1], size=x_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 10.0], y0=x[:,0], method='RK45')
    initial_xs[episode] = soln.y[:,-1]

for episode in range(num_episodes):
    x = np.vstack(initial_xs[episode])
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        u = np.random.choice(all_us, size=u_column_shape)
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

#%% Reward function
w_r = np.array([
    [0],
    [0],
    [1]
])
Q_ = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])
R = 0.0001
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat # (xs.shape[1], us.shape[1])
def reward(x, u):
    return -cost(x, u)

#%% Estimate Q function for current policy
w_hat_batch_size = 2**14 # 2**9 (4096)
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy_net.predict(x_batch.T) # (all_us.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_us.data.numpy()])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = phi_x_prime_batch * \
                                pi_response.reshape(
                                    pi_response.shape[1],
                                    1,
                                    pi_response.shape[0]
                                ).data.numpy() # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = reward(x_batch, np.array([all_us.data.numpy()])) * pi_response.data.numpy() # (w_hat_batch_size, all_us.shape[0])
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    )

    return w_hat

def Q(x, u, w_hat_t):
    return reward(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u)

#%% Policy function as PyTorch model
class PolicyNetwork():
    def __init__(self, n_state, n_action, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state, n_action),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        """
            Compute the action probabilities of state s using the learning model
            @param s: input state array
            @return: predicted policy
        """
        return self.model(torch.Tensor(s))

    def update(self, returns, log_probs):
        """
            Update the weights of the policy network given the training samples
            @param returns: return (cumulative rewards) for each step in an episode
            @param log_probs: log probability for each step
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        """
            Estimate the policy and sample an action, compute its log probability
            @param s: input state
            @return: the selected action and log probability
        """
        probs = self.predict(s)
        action_index = torch.multinomial(probs, 1).item()
        action = all_us[action_index].item()
        log_prob = torch.log(probs[action_index])
        return action, log_prob

def reinforce(estimator, n_episode, gamma=1.0):
    """
        REINFORCE algorithm
        @param estimator: policy network
        @param n_episode: number of episodes
        @param gamma: the discount factor
    """
    for episode in range(n_episode):
        states = []
        actions = []
        log_probs = []
        rewards = []

        state = np.vstack(initial_xs[episode])

        # w_hat = w_hat_t()
        while len(rewards) < 10000:
            action, log_prob = estimator.get_action(state[:,0])
            next_state = tensor.f(state, action)
            earned_reward = reward(state, action)[0,0]

            states.append(state[:,0])
            actions.append(action)

            total_reward_episode[episode] += earned_reward
            log_probs.append(log_prob)
            rewards.append(earned_reward)

            if len(rewards) == 10000:
                returns = torch.zeros([len(rewards)])
                Gt = 0
                for i in range(len(rewards)-1, -1, -1):
                    Gt = rewards[i] + (gamma * Gt)
                    returns[i] = Gt
                    # Q_val = Q(
                    #     np.vstack(states[i]),
                    #     np.array([actions[i]]),
                    #     w_hat
                    # )
                    # returns[i] = Q_val[0,0]

                returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

                estimator.update(returns, log_probs)
                if episode == 0 or (episode+1) % 100 == 0:
                    print(f"Episode: {episode+1}, total reward: {total_reward_episode[episode]}")
                    torch.save(estimator, PATH)

                break

            state = next_state

#%%
# n_state = env.observation_space.shape[0]
num_actions = all_us.shape[0]
lr = 0.003
# policy_net = PolicyNetwork(x_dim, num_actions, lr)
# def init_weights(m):
#     if type(m) == torch.nn.Linear:
#         m.weight.data.fill_(0.0)
# policy_net.model.apply(init_weights)
policy_net = torch.load(PATH)

num_episodes = 2000
gamma = 0.99
total_reward_episode = [0] * num_episodes

initial_xs = np.zeros([num_episodes, x_dim])
for episode in range(num_episodes):
    x = np.random.random(x_column_shape) * 0.5 * np.random.choice([-1,1], size=x_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 10.0], y0=x[:,0], method='RK45')
    initial_xs[episode] = soln.y[:,-1]

#%% Run REINFORCE
reinforce(policy_net, num_episodes, gamma)

# import matplotlib.pyplot as plt
# plt.plot(total_reward_episode)
# plt.title('Episode reward over time')
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()

#%% Test policy in environment
num_episodes = 1000

initial_xs = np.zeros([num_episodes, x_dim])
for episode in range(num_episodes):
    x = np.random.random(x_column_shape) * 0.5 * np.random.choice([-1,1], size=x_column_shape)
    u = np.array([[0]])
    soln = solve_ivp(fun=continuous_f(u), t_span=[0, 10.0], y0=x[:,0], method='RK45')
    initial_xs[episode] = soln.y[:,-1]

def watch_agent():
    costs = torch.zeros([num_episodes])
    for episode in range(num_episodes):
        state = np.vstack(initial_xs[episode])
        cumulative_cost = 0
        step = 0
        while step < 10000:
            # env.render()
            with torch.no_grad():
                action, _ = policy_net.get_action(state[:,0])
            state = tensor.f(action, state)
            cumulative_cost += cost(state, action)
            step += 1
            if step == 10000:
                costs[episode] = cumulative_cost
                # print(f"Total cost for episode {episode}:", cumulative_cost)
    # env.close()
    print(f"Mean cost per episode over {num_episodes} episodes:", torch.mean(costs))
watch_agent()

#%%