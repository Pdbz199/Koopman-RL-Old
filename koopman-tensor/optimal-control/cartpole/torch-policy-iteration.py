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
# import cartpole_reward
import observables

#%% Initialize environment
# env_string = 'CartPole-v0'
env_string = 'env:CartPoleControlEnv-v0'
env = gym.make(env_string)

# def reward(xs, us):
#     return cartpole_reward.defaultCartpoleRewardMatrix(xs, us).T
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
        Y[:,i], _, done, __ = env.step( int(U[0,i]) if env_string == 'CartPole-v0' else np.array([int(U[0,i])]) )
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
step_size = 1 if env_string == 'CartPole-v0' else 0.1
u_range = np.array([0, 1+step_size]) if env_string == 'CartPole-v0' else np.array([-5, 5+step_size])
all_us = np.arange(u_range[0], u_range[1], step_size)
all_us = np.round(all_us, decimals=2)

# HIDDEN_SIZE = 256

# Other NN spec:
# model = torch.nn.Sequential(
#     torch.nn.Linear(obs_size, HIDDEN_SIZE),
#     torch.nn.ReLU(),
#     torch.nn.Linear(HIDDEN_SIZE, n_actions),
#     torch.nn.Softmax(dim=0)
# )
model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, all_us.shape[0]),
    torch.nn.Softmax(dim=0)
)
# print("Model:", model)

learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Horizon = 500
MAX_TRAJECTORIES = 10000 # 4000 with traditional REINFORCE | 500, 750 with other NN spec
gamma = 0.99
score = []

w_hat_batch_size = 2**12 # 4096
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices]
    phi_x_batch = tensor.phi(x_batch)

    pi_response = model(torch.from_numpy(phi_x_batch.T).float()) # (w_hat_batch_size, all_us.shape[0])

    phi_x_primes = tensor.K_(np.array([all_us])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)

    expectation_term_1 = torch.sum(
        torch.from_numpy(phi_x_primes) * pi_response.reshape( pi_response.shape[1],1,pi_response.shape[0] ),
        dim=0
    ) # (phi_dim, w_hat_batch_size)

    costs = -cost(x_batch, all_us) * pi_response.T.data.numpy() # (all_us.shape[0], w_hat_batch_size)
    expectation_term_2 = torch.sum(torch.from_numpy(costs), dim=0) # (w_hat_batch_size,)

    return OLS(
        (phi_x_batch - gamma*expectation_term_1.data.numpy()).T,
        np.array([expectation_term_2.data.numpy()]).T
    )

def Q(x, u, w_hat_t):
    return -cost(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u)

for trajectory in range(MAX_TRAJECTORIES):
    curr_state = np.vstack(env.reset())
    done = False
    transitions = []

    for t in range(Horizon):
        phi_curr_state = tensor.phi(curr_state)
        act_prob = model(torch.from_numpy(phi_curr_state[:,0]).float())
        action = np.random.choice(all_us, p=act_prob.data.numpy())
        prev_state = curr_state[:,0]
        curr_state, _, done, info = env.step( action if env_string == 'CartPole-v0' else np.array([action]) )
        curr_state = np.vstack(curr_state)
        done = bool(
            curr_state[0,0] < -env.x_threshold
            or curr_state[0,0] > env.x_threshold
            or curr_state[2,0] < -env.theta_threshold_radians
            or curr_state[2,0] > env.theta_threshold_radians
        )
        transitions.append((prev_state, action, t+1))
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
        new_Gval = 0
        power = 0
        for j in range(i, len(transitions)):
            discount = gamma**power
            new_Gval = new_Gval + ( discount * -cost( np.vstack(transitions[j][0]), np.array([[transitions[j][1]]]) ) ) # reward_batch[j] # .numpy()
            power += 1

        Q_val = Q( np.vstack(state_batch[:,i]), np.array([[action_batch[i]]]), w_hat )[0,0]

        # batch_Gvals.append( new_Gval )
        batch_Gvals.append( Q_val )

        errors.append( np.abs(Q_val - new_Gval) )
    expected_returns_batch = torch.FloatTensor(batch_Gvals) # (batch_size,)
    expected_returns_batch /= expected_returns_batch.max()

    pred_batch = model(torch.from_numpy(tensor.phi(state_batch.data.numpy())).float().T) # (batch_size, num_actions)
    # prob_batch = pred_batch.gather(dim=1, index=(action_batch.long().view(-1,1))).squeeze() # (batch_size,)
    action_batch_indices = []
    for action in action_batch.data.numpy():
        rounded_action = np.round(action)
        action_batch_indices.append(np.where(all_us == rounded_action)[0][0])
    action_batch_indices = torch.from_numpy(np.array([action_batch_indices]))
    prob_batch = pred_batch.gather(dim=1, index=action_batch_indices).squeeze() # (batch_size,)

    loss = -torch.sum(torch.log(prob_batch) * expected_returns_batch)

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
        print("Average error between Q and G along trajectory:", np.mean(errors))

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
        episode_rewards = []
        while not done:
            env.render()
            phi_state = tensor.phi(state)
            pred = model(torch.from_numpy(phi_state[:,0]).float())
            action = np.random.choice(all_us, p=pred.data.numpy())
            state, reward, done, _ = env.step( action if env_string == 'CartPole-v0' else np.array([action]) )
            state = np.vstack(state)
            episode_rewards.append(reward)
            if done:
                rewards[episode] = np.sum(episode_rewards)
                print("Reward:", rewards[episode])
    env.close()
    print(f"Mean reward per episode over {num_episodes} episodes:", np.mean(rewards))
watch_agent()