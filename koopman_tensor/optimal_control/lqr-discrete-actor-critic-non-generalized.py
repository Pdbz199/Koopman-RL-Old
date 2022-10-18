# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from control import dare
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor, OLS
sys.path.append('../../')
import observables

# Configuration parameters for the whole setup
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

#%% Initialize environment
epsilon = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

state_dim = 3
action_dim = 1

A = np.zeros([state_dim, state_dim])
max_abs_real_eigen_val = 1.0
while max_abs_real_eigen_val >= 1.0 or max_abs_real_eigen_val <= 0.7:
    Z = np.random.rand(*A.shape)
    _,sigma,__ = np.linalg.svd(Z)
    Z /= np.max(sigma)
    A = Z.T @ Z
    W,_ = np.linalg.eig(A)
    max_abs_real_eigen_val = np.max(np.abs(np.real(W)))

print("A:", A)
print("A's max absolute real eigenvalue:", max_abs_real_eigen_val)
B = np.ones([state_dim,action_dim])

def f(x, u):
    return A @ x + B @ u

#%% Define cost
Q = np.eye(state_dim)
R = 1
w_r = np.array([
    [0.0],
    [0.0],
    [0.0]
])
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns are snapshots
    # x.T Q x + u.T R u
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.power(u, 2)*R
    return mat.T
def reward(x, u):
    return -cost(x, u)

#%% Initialize important vars
# state_range = 25.0
state_range = 5.0
state_minimums = np.ones([state_dim,1]) * -state_range
state_maximums = np.ones([state_dim,1]) * state_range

# action_range = 75.0
action_range = 20.0
# action_range = 10.0
action_minimums = np.ones([action_dim,1]) * -action_range
action_maximums = np.ones([action_dim,1]) * action_range

state_order = 2
action_order = 2

phi_dim = int( comb( state_order+state_dim, state_order ) )

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

# step_size = 0.1
step_size = 1.0
all_actions = np.arange(-action_range, action_range+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

dt = 1
gamma = 0.99
reg_lambda = 1.0

#%% Default policies
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

#%% Optimal policy
P = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q, R)[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = reg_lambda * np.linalg.inv(R + B.T @ P @ B)

def lqr_policy(x):
    return np.random.normal(-C @ (x - w_r), sigma_t)

#%% Construct datasets
num_episodes = 500
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, [state_dim,N])
U = np.random.uniform(action_minimums, action_maximums, [action_dim,N])
Y = f(X, U)

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
#     nn.Linear(state_dim, 128),
#     nn.ReLU(),
#     nn.Linear(128, 256),
#     nn.ReLU(),
#     nn.Linear(256, len(all_actions)),
#     nn.Softmax(dim=-1)
# )
# critic = nn.Sequential(
#     nn.Linear(phi_dim, 128),
#     nn.ReLU(),
#     nn.Linear(128, 256),
#     nn.ReLU(),
#     nn.Linear(256, 1)
# )

policy = nn.Sequential(
    nn.Linear(state_dim, len(all_actions)),
    nn.Softmax(dim=-1)
)
# critic = torch.zeros([1, phi_dim], requires_grad=True)
critic = np.zeros([1, phi_dim])

policy_learning_rate = 0.0003
critic_learning_rate = 0.0003

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)
# critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)
# critic_optimizer = torch.optim.Adam([critic], lr=critic_learning_rate)

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
        reward(x_batch, np.array([all_actions])),
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
running_reward = 0
episode_count = 0
# max_steps_per_episode = 200
max_steps_per_episode = 100

while True:  # Run until solved
    state = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim,1]
    )[:,0]
    episode_reward = 0

    states = []
    actions = []
    log_probs = []
    critic_values = []
    rewards = []
    for timestep in range(1, max_steps_per_episode):
        state = torch.Tensor(state)
        states.append(state.numpy())

        # Predict action probabilities and estimated future rewards from current state
        action_probs = policy(state)
        # critic_value = critic(torch.Tensor(tensor.phi(np.vstack(state))[:,0]))
        # critic_value = critic @ torch.Tensor(tensor.phi(np.vstack(state)))
        critic_value = torch.Tensor(critic @ tensor.phi(np.vstack(state)))
        critic_values.append(critic_value)

        # Sample action from action probability distribution
        action_index = np.random.choice(len(all_actions), p=np.squeeze(action_probs.detach().numpy()))
        action = all_actions[action_index]
        actions.append(action)
        action = np.array([[action]])
        log_probs.append(torch.log(action_probs[action_index]))

        # Calculate reward for current state/action pair
        curr_reward = reward(np.vstack(state), action)[0,0]
        rewards.append(curr_reward)
        episode_reward += curr_reward

        # Apply the sampled action in our environment
        state = f(np.vstack(state), action)[:,0]

        # Check done condition
        # done = np.linalg.norm(state) > 1000

        # if done:
        #     break

    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0
    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + epsilon)
    returns = returns.tolist()

    # Calculating loss values to update our network
    actor_losses = []
    critic_losses = []
    for log_prob, value, ret in zip(log_probs, critic_values, returns):
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
    # critic_optimizer.zero_grad()
    loss_value.backward()
    policy_optimizer.step()
    # critic_optimizer.step()
    critic = update_w_hat(2**12)

    # Log details
    episode_count += 1
    if episode_count % 100 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if episode_reward > -2_000:
    # if running_reward > -10_000_000:
        states = np.array(states)
        actions = np.array(actions)

        print(f"Learned by episode {episode_count}")

        print(f"Reward for final episode (Koopman controller): {np.sum(rewards)}\n")

        print(f"Initial state of final episode (Koopman controller): {states[0]}")
        print(f"Final state of final episode (Koopman controller): {states[-1]}\n")

        print(f"Reference state: {w_r[:,0]}\n")

        print(f"Difference between final state of final episode and reference state (Koopman controller): {np.abs(states[-1] - w_r[:,0])}")
        print(f"Norm between final state of final episode and reference state (Koopman controller): {np.linalg.norm(states[-1] - w_r[:,0])}")

        plt.title("Dynamics Over Time (Koopman Controller)")
        plt.xlabel("Timestep")
        plt.ylabel("State value")

        labels = []
        for i in range(state_dim):
            labels.append(f"x_{i}")
        for i in range(state_dim):
            plt.plot(states[:,i], label=labels[i])
        plt.legend(labels)

        plt.tight_layout()
        plt.show()

        plt.plot(states[:,0], states[:,1])
        plt.title("Koopman Controller in Environment (2D)")
        plt.show()

        labels = ['Koopman controller']

        plt.hist(actions)
        plt.legend(labels)
        plt.show()

        plt.scatter(np.arange(len(actions)), actions, s=5)
        plt.legend(labels)
        plt.show()

        break