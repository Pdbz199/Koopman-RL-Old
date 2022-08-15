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
state_order = 2
action_order = 2
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

# obs_size = env.observation_space.shape[0]
state_dim = env.observation_space.shape[0]
action_dim = 1

state_M_plus_N_minus_ones = np.arange( (state_dim-1), state_order + (state_dim-1) + 1 )
phi_dim = int( np.sum( comb( state_M_plus_N_minus_ones, np.ones_like(state_M_plus_N_minus_ones) * (state_dim-1) ) ) )

action_M_plus_N_minus_ones = np.arange( (action_dim-1), action_order + (action_dim-1) + 1 )
psi_dim = int( np.sum( comb( action_M_plus_N_minus_ones, np.ones_like(action_M_plus_N_minus_ones) * (action_dim-1) ) ) )

step_size = 1 if env_string == 'CartPole-v0' else 3.0 #0.1
u_range = np.array([0, 1+step_size]) if env_string == 'CartPole-v0' else np.array([-15, 15+step_size])
all_us = np.arange(u_range[0], u_range[1], step_size)
all_us = np.round(all_us, decimals=2)

# Model spec
policy_model = torch.nn.Sequential(
    torch.nn.Linear(phi_dim+psi_dim, action_dim)
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

w_hat_batch_size = X.shape[1] # 2**14 # 2**9 (4096)
def w_hat_t():
    x_batch_indices = np.random.choice(X.shape[1], w_hat_batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # (x_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy_model(torch.from_numpy(x_batch).float().T).T # (all_us.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_us])) @ phi_x_batch # (all_us.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = phi_x_prime_batch * \
                                pi_response.reshape(
                                    pi_response.shape[0],
                                    1,
                                    pi_response.shape[1]
                                ).data.numpy() # (all_us.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    cost_batch_prob = cost(x_batch, np.array([all_us])) * pi_response.data.numpy() # (all_us.shape[0], w_hat_batch_size)
    expectation_term_2 = np.array([
        np.sum(cost_batch_prob, axis=0) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    w_hat = OLS(
        (phi_x_batch - (gamma*expectation_term_1)).T,
        expectation_term_2.T
    )

    return w_hat

def Q(x, u, w_hat_t):
    return cost(x, u) + gamma*w_hat_t.T @ tensor.phi_f(x, u)

for trajectory in range(MAX_TRAJECTORIES):
    curr_state = np.vstack(env.reset())
    done = False
    transitions = []

    total_probability = 0
    for t in range(Horizon):
        phi_curr_state = tensor.phi(curr_state)
        act_probs = torch.zeros([all_us.shape[0]])
        for i in range(all_us.shape[0]):
            with torch.no_grad():
                act_prob = policy_model(torch.from_numpy(
                    np.append(
                        phi_curr_state[:,0],
                        tensor.psi(np.array([[ all_us[i] ]]))
                    )
                ).float())
            act_probs[i] = act_prob
        total_probability = torch.sum(torch.exp(act_probs))
        act_probs = torch.exp(act_probs) / total_probability # Softmax
        # print("act_probs:", act_probs)
        u = np.random.choice(all_us, p=act_probs.data.numpy())
        action = u if env_string == 'CartPole-v0' else np.array([u])
        prev_state = curr_state[:,0]
        curr_state, curr_reward, done, _ = env.step(action)
        curr_state = np.vstack(curr_state)
        # cumulative_reward = 0 if t == 0 else transitions[-1][2]
        print(curr_reward)
        transitions.append((prev_state, u, curr_reward))
        if done:
            break
    score.append(len(transitions))

    state_batch = torch.from_numpy(np.array([s for (s,_,__) in transitions])).float().T # (state_dim, len(transitions))
    action_batch = torch.Tensor([a for (_,a,__) in transitions]) # (len(transitions),)
    reward_batch = torch.Tensor([r for (_,__,r) in transitions])#.flip(dims=(0,)) # (len(transitions),)

    batch_Gvals = []
    errors = []
    # w_hat = w_hat_t()
    for t in range(len(transitions)):
        new_Gval = 0
        power = 0
        for k in range(t, len(transitions)):
            discount = gamma**power
            new_Gval += discount * reward_batch[k]
            power += 1

        # Q_val = Q(
        #     np.vstack(state_batch[:,i]),
        #     np.array([[action_batch[i]]]),
        #     w_hat
        # )[0,0]

        batch_Gvals.append( new_Gval )
        # batch_Gvals.append( Q_val )
        # print("Q:", Q_val)
        # print("G:", new_Gval)

        # errors.append( np.abs( Q_val - new_Gval ) )
        # print("Error:", errors[-1])
    Gval_batch = torch.FloatTensor(batch_Gvals) # (batch_size,)
    # Gval_batch /= Gval_batch.max()
    Gval_batch /= torch.abs(Gval_batch.max())

    act_prob_batch = torch.zeros([len(transitions)])
    for i in range(len(transitions)):
        act_prob = policy_model(torch.from_numpy(
            np.append(
                tensor.phi(np.vstack(state_batch[:,i].data.numpy())),
                tensor.psi(np.array([[action_batch[i].data.numpy()]]))
            )
        ).float())
        act_prob_batch[i] = torch.exp(act_prob) / total_probability
        
    # action_batch_indices = []
    # for action in action_batch.data.numpy():
    #     rounded_action = np.round(action, decimals=2)
    #     action_batch_indices.append(np.where(all_us == rounded_action)[0])
    # action_batch_indices = torch.from_numpy(np.array(action_batch_indices))
    # What is gather? https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    # prob_batch = pred_batch.gather(dim=1, index=action_batch_indices).squeeze() # (batch_size,)

    # Compute loss

    # Original code
    # loss = torch.sum(-act_prob_batch * Gval_batch)

    # Found this version
    log_prob = -torch.log(act_prob_batch)
    selected_logprobs = log_prob * Gval_batch
    loss = torch.mean(selected_logprobs)

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