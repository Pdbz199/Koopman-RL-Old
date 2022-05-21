import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

from control import dare
from scipy.special import comb
# from scipy.integrate import quad_vec

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../')
import cartpole_reward
import observables
import utilities

#%% Initialize environment
state_dim = 3
action_dim = 1

A_shape = [state_dim,state_dim]
A = np.zeros(A_shape)
max_real_eigen_val = 1.0
while max_real_eigen_val >= 1.0 or max_real_eigen_val <= 0.7:
    Z = np.random.rand(*A_shape)
    A = Z.T @ Z
    W,V = np.linalg.eig(A)
    max_real_eigen_val = np.max(np.real(W))
print("A:", A)
print("A's max real eigenvalue:", max_real_eigen_val)
B = np.ones([A_shape[0],action_dim])

def f(x, u):
    return A @ x + B @ u

#%% Define cost
Q_ = np.eye(A.shape[1])
R = 1
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    # x.T Q x + u.T R u
    mat = np.vstack(np.diag(x.T @ Q_ @ x)) + np.power(u, 2)*R
    return mat.T
def reward(x, u):
    return -cost(x, u)

#%% Reward function
# def reward(xs, us):
#     return cartpole_reward.defaultCartpoleRewardMatrix(xs, np.array([us])).T
# def cost(xs, us):
#     return -reward(xs, us)

#%% Initialize important vars
state_range = 25
action_range = 25
state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]
phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

step_size = 0.01
all_us = torch.arange(-5, 5+step_size, step_size) # -20 to 20
all_us = np.round(all_us, decimals=2)

gamma = 0.99
lamb = 0.0001

#%% Optimal policy
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q_, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)
def optimal_policy(x):
    return np.random.normal(-(C @ x), sigma_t)

#%% Construct datasets
num_episodes = 100
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Path-based approach
# X = np.zeros([state_dim,N])
# U = np.zeros([action_dim,N])
# Y = np.zeros([state_dim,N])
# for episode in range(num_episodes):
#     state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
#     done = False
#     step = 0
#     for step in range(num_steps_per_episode):
#         X[:,(episode*num_steps_per_episode)+step] = state[:,0]
#         # u = np.random.choice(all_us)
#         u = optimal_policy(state)[0,0]
#         U[0,(episode*num_steps_per_episode)+step] = u
#         Y[:,(episode*num_steps_per_episode)+step] = f(state, np.array([[ u ]]))[:, 0]
#         state = np.vstack(Y[:,(episode*num_steps_per_episode)+step])

# Shotgun-based approach
X = np.random.rand(A.shape[0],N)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],N))
U = np.random.rand(B.shape[1],N)*action_range*np.random.choice(np.array([-1,1]), size=(B.shape[1],N))
# U = optimal_policy(X)
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

#%% Shotgun-based training error
training_norms = np.zeros([X.shape[1]])
state_norms = np.zeros([X.shape[1]])
for i in range(X.shape[1]):
    state = np.vstack(X[:,i])
    state_norms[i] = utilities.l2_norm(state, np.zeros_like(state))
    action = np.vstack(U[:,i])
    true_x_prime = np.vstack(Y[:,i])
    predicted_x_prime = tensor.f(state, action)
    training_norms[i] = utilities.l2_norm(true_x_prime, predicted_x_prime)
average_training_norm = np.mean(training_norms)
average_state_norm = np.mean(state_norms)
print(f"Average training norm: {average_training_norm}")
print(f"Average training norm normalized by average state norm: {average_training_norm / average_state_norm}")

#%% Path-based training error
# training_norms = np.zeros([num_episodes,num_steps_per_episode])
# state_norms = np.zeros([X.shape[1]])
# for episode in range(num_episodes):
#     for step in range(num_steps_per_episode):
#         state = np.vstack(X[:,(episode*num_steps_per_episode)+step])
#         state_norms[(episode*num_steps_per_episode)+step] = utilities.l2_norm(state, np.zeros_like(state))
#         action = np.vstack(U[:,(episode*num_steps_per_episode)+step])
#         true_x_prime = np.vstack(Y[:,(episode*num_steps_per_episode)+step])
#         predicted_x_prime = tensor.f(state, action)
#         training_norms[episode,step] = utilities.l2_norm(true_x_prime, predicted_x_prime)
#         state = true_x_prime
# average_training_norm_per_episode = np.mean(np.sum(training_norms, axis=1))
# average_state_norm = np.mean(state_norms)
# print(f"Average training norm per episode over {num_episodes} episodes: {average_training_norm_per_episode}")
# print(f"Average training norm per episode over {num_episodes} episodes normalized by average state norm: {average_training_norm_per_episode / average_state_norm}")

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
        log_prob = torch.log(probs[action_index])
        action = all_us[action_index].item()
        return action, log_prob

def reinforce(estimator, n_episode, gamma=1.0):
    """
        REINFORCE algorithm
        @param estimator: policy network
        @param n_episode: number of episodes
        @param gamma: the discount factor
    """
    w_hat = np.zeros([phi_dim,1])
    for episode in range(n_episode):
        states = []
        actions = []
        log_probs = []
        rewards = []
        state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))

        while len(rewards) < 300:
            u, log_prob = estimator.get_action(state[:,0])
            action = np.array([[u]])
            next_state = f(state, action)
            curr_reward = reward(state, action)[0,0]

            states.append(state)
            actions.append(u)

            total_reward_episode[episode] += curr_reward
            log_probs.append(log_prob)
            rewards.append(curr_reward)

            if len(rewards) == 300:
                returns = torch.zeros([len(rewards)])
                # Gt = 0
                for i in range(len(rewards)-1, -1, -1):
                    # Gt = rewards[i] + (gamma * Gt)
                    # returns[i] = Gt
                    # print("Gt:", Gt)
                    Q_val = Q(
                        np.vstack(states[i]),
                        np.array([[actions[i]]]),
                        w_hat
                    )
                    returns[i] = Q_val
                    # print("Q_val:", Q_val)

                # if episode == 0 or (episode+1) % 100 == 0:
                #     print(returns)

                returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float64).eps)

                estimator.update(returns, log_probs)
                if episode == 0 or (episode+1) % 100 == 0:
                    print(f"Episode: {episode+1}, total reward: {total_reward_episode[episode]}")
                    torch.save(estimator, 'lqr-policy-model.pt')

                break

            state = next_state

        w_hat = w_hat_t()

#%%
n_state = A.shape[0]
n_action = all_us.shape[0]
lr = 0.003

def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

policy_net = PolicyNetwork(n_state, n_action, lr)
policy_net.model.apply(init_weights)

# policy_net = torch.load('lqr-policy-model.pt')

n_episode = 2000
gamma = 0.99
lamb = 0.0001
total_reward_episode = [0] * n_episode

#%% Estimate Q function for current policy
w_hat_batch_size = 2**11 # 2**14
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy_net.predict(x_batch.T).T # (all_us.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_us.data.numpy()])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response.data.numpy()) # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = np.einsum('uw,uw->wu', reward(x_batch, np.array([all_us.data.numpy()])), pi_response.data.numpy()) # (w_hat_batch_size, all_us.shape[0])
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

#%% REINFORCE
reinforce(policy_net, n_episode, gamma)

# import matplotlib.pyplot as plt
# plt.plot(total_reward_episode)
# plt.title('Episode reward over time')
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()

test_steps = 200
def watch_agent():
    optimal_states = np.zeros([test_steps,n_state])
    learned_states = np.zeros([test_steps,n_state])

    for episode in range(num_episodes):
        state = np.random.rand(A.shape[0],1)*state_range*np.random.choice(np.array([-1,1]), size=(A.shape[0],1))
        optimal_state = state
        learned_state = state
        step = 0
        while step < test_steps:
            optimal_states[step] = optimal_state[:,0]
            learned_states[step] = learned_state[:,0]

            optimal_action = np.random.normal(-(C @ optimal_state), sigma_t)
            optimal_state = f(optimal_state, optimal_action)

            with torch.no_grad():
                learned_u, _ = policy_net.get_action(learned_state[:,0])
            learned_action = np.array([[learned_u]])
            learned_state = f(learned_state, learned_action)

            step += 1

    print("Norm between entire paths:", utilities.l2_norm(optimal_states, learned_states))

    fig, axs = plt.subplots(2)
    fig.suptitle('Dynamics Over Time')

    axs[0].set_title('True dynamics')
    axs[0].set(xlabel='Timestep', ylabel='State value')

    axs[1].set_title('Learned dynamics')
    axs[1].set(xlabel='Timestep', ylabel='State value')

    labels = np.array(['x_0', 'x_1', 'x_2', 'x_3'])
    for i in range(A.shape[1]):
        axs[0].plot(optimal_states[:,i], label=labels[i])
        axs[1].plot(learned_states[:,i], label=labels[i])
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    plt.tight_layout()
    plt.show()

watch_agent()

#%%