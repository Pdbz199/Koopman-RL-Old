#%% Imports
import gym
import matplotlib.pyplot as plt
import numpy as np

from control import dare, dlqr

seed = 123
np.random.seed(seed)

#%% Gym environment
env = gym.make('env:CartPoleControlEnv-v0')

#%% System dynamics
A = np.array([
    [1.0, 0.02,  0.0,        0.0 ],
    [0.0, 1.0,  -0.01434146, 0.0 ],
    [0.0, 0.0,   1.0,        0.02],
    [0.0, 0.0,   0.3155122,  1.0 ]
])
B = np.array([
    [0],
    [0.0195122],
    [0],
    [-0.02926829]
])

state_dim = A.shape[1]
action_dim = B.shape[1]

action_range = 25

gamma = 0.99
lamb = 0.0001

def f(x, u):
    return A @ x + B @ u

theta_threshold_radians = 12 * 2 * np.pi / 360
x_threshold = 2.4
def is_done(state):
    """
        INPUTS:
        state - state array
    """
    return bool(
        state[0] < -x_threshold
        or state[0] > x_threshold
        or state[2] < -theta_threshold_radians
        or state[2] > theta_threshold_radians
    )

#%% Cost
Q_ = np.array([
    [10.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, 10.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0]
])
R = 0.1
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    # x.T Q x + u.T R u
    mat = np.vstack(np.diag(x.T @ Q_ @ x)) + np.power(u, 2)*R
    return mat.T
def reward(x, u):
    return -cost(x, u)

#%% Random policy
def random_policy(state=None):
    return np.random.rand(action_dim,1)*action_range*np.random.choice(np.array([-1,1]), size=(action_dim,1))

#%% Optimal policy
soln = dare(A*np.sqrt(gamma), B*np.sqrt(gamma), Q_, R)
P = soln[0]
C = np.linalg.inv(R + gamma*B.T @ P @ B) @ (gamma*B.T @ P @ A)
sigma_t = lamb * np.linalg.inv(R + B.T @ P @ B)
def optimal_policy(x):
    return np.random.normal(-(C @ x), sigma_t)

# C = dlqr(A, B, Q_, R)[0]
# def optimal_policy(x):
#     return -(C @ x)

#%% When explosion?
num_episodes = 200
num_steps_per_episode = 500
env_states_over_time = np.zeros([num_episodes, num_steps_per_episode, state_dim])
env_costs_over_time = np.zeros([num_episodes, num_steps_per_episode])
f_states_over_time = np.zeros([num_episodes, num_steps_per_episode, state_dim])
f_costs_over_time = np.zeros([num_episodes, num_steps_per_episode])
num_dones = 0
for episode in range(num_episodes):
    state = env.reset()
    env_state = state
    f_state = state
    for step in range(num_steps_per_episode):
        env_states_over_time[episode,step] = env_state
        f_states_over_time[episode,step] = f_state

        action = random_policy()
        # action = optimal_policy(env_state)
        # action = optimal_policy(f_state)

        env_state, _, __, ___ = env.step(action[0])
        f_state = f(np.vstack(state), action)[:,0]

        env_costs_over_time[episode,step] = cost(np.vstack(env_state), action)
        f_costs_over_time[episode,step] = cost(np.vstack(f_state), action)

        if is_done(env_state) or is_done(f_state):
            num_dones += 1
print(f"Average num dones: {num_dones/num_episodes}")

#%% Plot vars
state_index_to_plot = 5
labels = np.array(['cart position', 'cart velocity', 'pole angle', 'pole angular velocity'])

#%% Plot costs over time
fig, axs = plt.subplots(2)
fig.suptitle('Cost Over Time')

axs[0].set_title('Env cost')
axs[0].set(xlabel='Timestep', ylabel='Cost value')

axs[1].set_title('LQR cost')
axs[1].set(xlabel='Timestep', ylabel='Cost value')

axs[0].plot(env_costs_over_time[state_index_to_plot])
axs[1].plot(f_costs_over_time[state_index_to_plot])

plt.tight_layout()
plt.show()

#%% Plot states over time
fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')

axs[0].set_title('True dynamics')
axs[0].set(xlabel='Timestep', ylabel='State value')

axs[1].set_title('LQR dynamics')
axs[1].set(xlabel='Timestep', ylabel='State value')

for i in range(A.shape[1]):
    axs[0].plot(env_states_over_time[state_index_to_plot,:,i], label=labels[i])
    axs[1].plot(f_states_over_time[state_index_to_plot,:,i], label=labels[i])
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()

#%%