import gym
import numpy as np
import random
import torch

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

from matplotlib import pyplot as plt
from scipy.special import comb

import sys
sys.path.append('../../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../../')
import cartpole_reward
import observables
import utilities

#%% Initialize environment
env_string = 'CartPole-v0'
# env_string = 'env:CartPoleControlEnv-v0'
env = gym.make(env_string)

# def reward(xs, us):
#     return cartpole_reward.defaultCartpoleRewardMatrix(xs, us)
# def cost(xs, us):
#     return -reward(xs, us)

w_r = np.zeros([4,1])
Q_ = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    return mat.T
def reward(x, u):
    return -cost(x, u)

#%% Construct snapshots of u from random agent and initial states x0
N = 20000 # Number of datapoints
U = np.zeros([1,N])
X = np.zeros([4,N+1])
Y = np.zeros([4,N])
i = 0
while i < N:
    X[:,i] = env.reset()
    done = False
    while i < N and not done:
        U[0,i] = env.action_space.sample()
        action = int(U[0,i]) if env_string == 'CartPole-v0' else np.array([int(U[0,i])])
        Y[:,i], _, done, __ = env.step(action)
        if not done:
            X[:,i+1] = Y[:,i]
        i += 1
X = X[:,:-1]

#%% Estimate Koopman tensor
order = 2
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(order),
    psi=observables.monomials(2),
    regressor='ols'
)

# obs_size = env.observation_space.shape[0]
state_dim = env.observation_space.shape[0]
M_plus_N_minus_ones = np.arange( (state_dim-1), order + (state_dim-1) + 1 )
phi_dim = int( np.sum( comb( M_plus_N_minus_ones, np.ones_like(M_plus_N_minus_ones) * (state_dim-1) ) ) )
step_size = 1 if env_string == 'CartPole-v0' else 3.0 #0.1
u_range = np.array([0, 1+step_size]) if env_string == 'CartPole-v0' else np.array([-15, 15+step_size])
all_us = np.arange(u_range[0], u_range[1], step_size)
all_us = np.round(all_us, decimals=2)

# Model spec
policy_model = torch.nn.Sequential(
    torch.nn.Linear(state_dim, all_us.shape[0]),
    torch.nn.Softmax(dim=-1) # was dim=0
)
# print("Policy model:", policy_model)
# Initialize weights with 0s
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)
policy_model.apply(init_weights)

learning_rate = 0.003
optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

Horizon = 500
# xxxx with REINFORCE on continuous action system | 4000 with traditional REINFORCE | 500, 750 with other NN spec
MAX_TRAJECTORIES = 10000
gamma = 0.99
score = []

w_hat_batch_size = 2**14
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy_model(torch.from_numpy(x_batch).float().T).T # (all_us.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_us])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    #! don't remember if checked properly

    phi_x_prime_batch_prob = phi_x_prime_batch * \
                                pi_response.reshape(
                                    pi_response.shape[0],
                                    1,
                                    pi_response.shape[1]
                                ).data.numpy() # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = reward(x_batch, np.array([all_us])) * pi_response.data.numpy() # (all_us.shape[0], w_hat_batch_size)
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=0) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    ) # (phi_dim, 1)

    return w_hat

def Q(x, u, w_hat_t):
    return reward(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u)

for trajectory in range(MAX_TRAJECTORIES):
    curr_state = np.vstack(env.reset())
    done = False
    transitions = []

    for t in range(Horizon):
        # phi_curr_state = tensor.phi(curr_state)
        with torch.no_grad():
            act_prob = policy_model(torch.from_numpy(curr_state[:,0]).float())
        # print("action probs across x's", act_prob)
        u = np.random.choice(all_us, p=act_prob.data.numpy())
        action = u if env_string == 'CartPole-v0' else np.array([u])
        prev_state = curr_state[:,0]
        curr_state, _, __, info = env.step(action)
        curr_state = np.vstack(curr_state)
        done = bool(
            curr_state[0,0] < -env.x_threshold
            or curr_state[0,0] > env.x_threshold
            or curr_state[2,0] < -env.theta_threshold_radians
            or curr_state[2,0] > env.theta_threshold_radians
        )
        transitions.append((prev_state, u, t+1))
        if done:
            break
    score.append(len(transitions))

    state_batch = torch.from_numpy(np.array([s for (s,a,r) in transitions]).T)
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))

    batch_Gvals = []
    errors = []
    w_hat = w_hat_t()
    for i in range(len(transitions)):
        # new_Gval = 0
        # power = 0
        # for j in range(i, len(transitions)):
        #     discount = gamma**power
        #     reward_value = reward(
        #         np.vstack( transitions[j][0] ),
        #         np.array([[ transitions[j][1] ]])
        #     )
        #     new_Gval = new_Gval + ( discount * reward_value[0,0] )
        #     power += 1

        Q_val = Q(
            np.vstack(state_batch[:,i]),
            np.array([[action_batch[i]]]),
            w_hat
        )[0,0]

        # batch_Gvals.append( new_Gval )
        batch_Gvals.append( Q_val )
        # print("Q:", Q_val)
        # print("G:", new_Gval)

        # errors.append( np.abs( Q_val - new_Gval ) )
        # print("Error:", errors[-1])
    expected_returns_batch = torch.FloatTensor(batch_Gvals) # (batch_size,)
    expected_returns_batch /= np.abs(expected_returns_batch.max())

    pred_batch = policy_model(state_batch.T.float()) # (batch_size, num_actions)
    action_batch_indices = []
    for action in action_batch.data.numpy():
        rounded_action = np.round(action, decimals=2)
        action_batch_indices.append(np.where(all_us == rounded_action)[0])
    action_batch_indices = torch.from_numpy(np.array(action_batch_indices))
    # What is gather? https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    prob_batch = pred_batch.gather(dim=1, index=action_batch_indices).squeeze() # (batch_size,)

    # Compute loss

    # Original code
    loss = torch.sum(-prob_batch * expected_returns_batch)

    # Found this version
    # logprob = torch.log(prob_batch)
    # selected_logprobs = logprob * expected_returns_batch
    # loss = -torch.sum(selected_logprobs)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Online learning
    # X = np.append(X, tensor.B.T @ tensor.phi(state_batch.data.numpy()), axis=1)
    # Y = np.roll(X, -1)[:,:-1]
    # X = X[:,:-1]
    # U = np.append(U, np.array([action_batch.data.numpy()]), axis=1)
    # tensor = KoopmanTensor(
    #     X,
    #     Y,
    #     U,
    #     phi=observables.monomials(order),
    #     psi=observables.monomials(order),
    #     regressor='ols'
    # )

    # Average score for trajectory
    if trajectory % 50 == 0 and trajectory>0:
        print(f'Trajectory {trajectory}\tAverage Score: {np.mean(score[-50:-1])}')
        # print("Average error between Q and G along trajectory:", np.mean(errors))

def running_mean(x):
    N = 50
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y
score = np.array(score)
avg_score = running_mean(score)
plt.figure(figsize=(15,7))
plt.ylabel("Trajectory Duration", fontsize=12)
plt.xlabel("Training Epochs", fontsize=12)
plt.plot(score, color='gray' , linewidth=1)
plt.plot(avg_score, color='blue', linewidth=3)
plt.scatter(np.arange(score.shape[0]), score, color='green' , linewidth=0.3)
plt.show()

num_episodes = 10
def watch_agent():
    rewards = np.zeros([num_episodes])
    for episode in range(num_episodes):
        state = np.vstack(env.reset())
        done = False
        step = 0
        while not done and step < 200:
            env.render()
            # phi_state = tensor.phi(state)
            with torch.no_grad():
                pred = policy_model(torch.from_numpy(state[:,0]).float())
            u = np.random.choice(all_us, p=pred.data.numpy())
            action = u if env_string == 'CartPole-v0' else np.array([u])
            state, _, done, __ = env.step(action)
            step += 1
            state = np.vstack(state)
            if done or step >= 200:
                rewards[episode] = step
                print("Reward:", step)
    env.close()
    print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))
watch_agent()