# Imports
import gym
import numpy as np
import torch
import torch.nn as nn

from scipy.special import comb

import sys
sys.path.append('../../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../../')
import observables
from cartpole_reward import defaultCartpoleRewardMatrix

# Configuration parameters for the whole setup
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

gamma = 0.99  # Discount factor for past rewards
dt = 1
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 2
num_hidden_1 = 128
num_hidden_2 = 256

state_order = 1
action_order = 1

phi_dim = int( comb( state_order+num_inputs, state_order ) )

# action_range = 75.0
# action_range = 20.0
# action_range = 10.0
# action_minimums = np.ones([num_actions,1]) * -action_range
# action_maximums = np.ones([num_actions,1]) * action_range

# step_size = 0.1
# step_size = 1.0
# all_actions = np.arange(-action_range, action_range+step_size, step_size)
# all_actions = np.round(all_actions, decimals=2)

all_actions = np.array([0,1])

#%% Define cost
# Q = np.eye(num_inputs)
# R = 1
# w_r = np.array([
#     [0.0],
#     [0.0],
#     [0.0]
# ])
# def cost(x, u):
#     # Assuming that data matrices are passed in for X and U. Columns are snapshots
#     # x.T Q x + u.T R u
#     x_ = x - w_r
#     mat = np.vstack(np.diag(x_.T @ Q @ x_)) + np.power(u, 2)*R
#     return mat.T

# Load training data
X = np.load('../../../random_agent/cartpole-states.npy').T
Y = np.append(np.roll(X, -1, axis=1)[:,:-1], np.zeros((X.shape[0],1)), axis=1)
U = np.load('../../../random_agent/cartpole-actions.npy').reshape(1,-1)

# X = np.load('../../../optimal_agent/cartpole-states.npy').T
# Y = np.append(np.roll(X, -1, axis=1)[:,:-1], np.zeros((X.shape[0],1)), axis=1)
# U = np.load('../../../optimal_agent/cartpole-actions.npy').reshape(1,-1)

# Train Koopman Tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

# Actor and critic models
# policy = nn.Sequential(
#     nn.Linear(num_inputs, num_hidden_1),
#     nn.ReLU(),
#     nn.Linear(num_hidden_1, num_hidden_2),
#     nn.ReLU(),
#     nn.Linear(num_hidden_2, num_actions),
#     nn.Softmax(dim=-1)
# )
# critic = nn.Sequential(
#     nn.Linear(num_inputs, num_hidden_1),
#     nn.ReLU(),
#     nn.Linear(num_hidden_1, num_hidden_2),
#     nn.ReLU(),
#     nn.Linear(num_hidden_2, 1)
# )
policy = nn.Sequential(
    nn.Linear(num_inputs, num_actions),
    nn.Softmax(dim=-1)
)
# critic = torch.zeros([1, num_inputs], requires_grad=True)
critic = torch.zeros([1, phi_dim], requires_grad=True)
# critic = np.zeros([1, phi_dim])

policy_learning_rate = 0.0003
critic_learning_rate = 0.0003

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)
critic_optimizer = torch.optim.Adam([critic], lr=critic_learning_rate)

# Update critic weights
def update_w_hat(w_hat_batch_size):
    x_batch_indices = np.random.choice(tensor.X.shape[1], w_hat_batch_size, replace=False)
    x_batch = tensor.X[:, x_batch_indices] # (state_dim, w_hat_batch_size)
    phi_x_batch = tensor.phi(x_batch) # (phi_dim, w_hat_batch_size)

    with torch.no_grad():
        pi_response = policy(torch.Tensor(x_batch.T)).T # (all_actions.shape[0], w_hat_batch_size)

    phi_x_prime_batch = tensor.K_(np.array([all_actions])) @ phi_x_batch # (all_actions.shape[0], phi_dim, w_hat_batch_size)
    phi_x_prime_batch_prob = np.einsum('upw,uw->upw', phi_x_prime_batch, pi_response.numpy()) # (all_actions.shape[0], phi_dim, w_hat_batch_size)
    expectation_term_1 = np.sum(phi_x_prime_batch_prob, axis=0) # (phi_dim, w_hat_batch_size)

    reward_batch_prob = np.einsum(
        'uw,uw->wu',
        defaultCartpoleRewardMatrix(x_batch, np.array([all_actions])),
        pi_response.data.numpy()
    ) # (w_hat_batch_size, all_actions.shape[0])
    expectation_term_2 = np.array([
        np.sum(reward_batch_prob, axis=1) # (w_hat_batch_size,)
    ]) # (1, w_hat_batch_size)

    return OLS(
        (phi_x_batch - ((gamma**dt)*expectation_term_1)).T,
        expectation_term_2.T
    ).T

# Other important vars
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):
        # if episode_count % 100 == 0:
        #     env.render()
        state = torch.Tensor(state)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs = policy(state)
        # critic_value = critic(state)
        # critic_value = torch.Tensor(critic @ tensor.phi(np.vstack(state)))
        critic_value = critic @ torch.Tensor(tensor.phi(np.vstack(state)))
        critic_value_history.append(critic_value)

        # Sample action from action probability distribution
        action = np.random.choice(num_actions, p=np.squeeze(action_probs.detach().numpy()))
        action_probs_history.append(torch.log(action_probs[action]))

        # Apply the sampled action in our environment
        state, reward, done, _ = env.step(action)
        rewards_history.append(reward)
        episode_reward += reward

        if done:
            break

    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
    returns = returns.tolist()

    # Calculating loss values to update our network
    actor_losses = []
    critic_losses = []
    for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
        # At this point in history, the critic estimated that we would get a
        # total reward = `value` in the future. We took an action with log probability
        # of `log_prob` and ended up recieving a total reward = `ret`.
        # The actor must be updated so that it predicts an action that leads to
        # high rewards (compared to critic's estimate) with high probability.
        diff = ret - value
        actor_losses.append(-log_prob * diff)  # actor loss

        # The critic must be updated so that it predicts a better estimate of
        # the future rewards.
        # critic_losses.append(
        #     huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
        # )
        critic_losses.append(torch.pow(ret - value, 2))

    # Compute loss
    loss_value = sum(actor_losses) + sum(critic_losses)

    # Backpropagation
    policy_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    loss_value.backward()
    policy_optimizer.step()
    critic_optimizer.step()
    # critic = update_w_hat(2**12)

    # Clear the loss and reward history
    action_probs_history.clear()
    critic_value_history.clear()
    rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 100 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break