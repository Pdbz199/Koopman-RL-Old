#%% Imports
import gym
import numpy as np
# from scipy.integrate import odeint

# from sklearn.kernel_approximation import RBFSampler

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import algorithmsv2_parallel as algorithmsv2
import observables
import utilities

#%% Load environment
env = gym.make('CartPole-v0')

#%% Define Q and R
# Q = np.eye(4)
# R = 0.0001

Q = np.array([
    [10, 0,  0, 0],
    [ 0, 1,  0, 0],
    [ 0, 0, 10, 0],
    [ 0, 0,  0, 1]
])
R = 0.1

#%% Construct datasets
# seed = 123
# env.seed(seed)
# np.random.seed(seed)

num_datapoints = 20000 # 50000

X = np.zeros([
    env.observation_space.sample().shape[0],
    num_datapoints
])
Y = np.zeros([
    env.observation_space.sample().shape[0],
    num_datapoints
])
U = np.zeros([
    1,
    num_datapoints
])

total_datapoints = 0
while total_datapoints < num_datapoints:
    x = env.reset()
    done = False
    while not done and total_datapoints < num_datapoints:
        X[:, total_datapoints] = x
        u = env.action_space.sample() # Sampled from random agent
        U[:, total_datapoints] = u
        y,reward,done,info = env.step(u)
        Y[:, total_datapoints] = y
        x = y
        total_datapoints += 1

# U[U == 1] = 10
# U[U == 0] = -10

#%% Tensor
# rbf_state_feature = RBFSampler(gamma=1, n_components=75, random_state=1)
# rbf_action_feature = RBFSampler(gamma=1, n_components=75, random_state=1)

# def phi(x):
#     """ x must be one or a set column vectors """
#     entry_0 = np.vstack(x[:,0])
#     assert entry_0.shape[0] >= entry_0.shape[1]
#     return rbf_state_feature.fit_transform(x.T).T

# def psi(u):
#     """ u must be one or a set of column vectors """
#     entry_0 = np.vstack(u[:,0])
#     assert entry_0.shape[0] >= entry_0.shape[1]
#     return rbf_action_feature.fit_transform(u.T).T

# gravity = 9.8
# masscart = 1.0
# masspole = 0.1
# total_mass = (masspole + masscart)
# length = 0.5 # actually half the pole's length
# polemass_length = (masspole * length)
# H = np.array([
# 	[1, 0, 0, 0],
# 	[0, total_mass, 0, - polemass_length],
# 	[0, 0, 1, 0],
# 	[0, - polemass_length, 0, (2 * length)**2 * masspole / 3]
# ])
# Hinv = np.linalg.inv(H)
# A = Hinv @ np.array([
#     [0, 1, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#     [0, 0, - polemass_length * gravity, 0]
# ])
# B = Hinv @ np.array([0, 1.0, 0, 0]).reshape([4, 1])

# mass_pole = 1
# mass_cart = 5
# pole_position = 1
# pole_length = 2
# gravity = -10
# cart_damping = 1
# A = np.array([
#     [0, 1, 0, 0],
#     [0, -cart_damping / mass_cart, pole_position * mass_pole * gravity / mass_cart, 0],
#     [0, 0, 0, 1],
#     [0, -pole_position * cart_damping / mass_cart * pole_length, -pole_position * (mass_pole + mass_cart) * gravity / mass_cart * pole_length]
# ]) #* From Databook V2
# B = np.array([
#     [0],
#     [1 / mass_cart],
#     [0],
#     [pole_position / mass_cart * pole_length]
# ])

# def f(x, u):
#     return A @ x + B @ u

def f(x, u):
    assert x.shape == (4,)
    assert u.shape == (1,)

    env = gym.make('CartPole-v0')
    env.reset(state = x)
    observation, cost, done, info = env.step(int(u[0]))
    env.close()

    return observation

def compute_A(x_s, u_s):
    return utilities.jacobian(lambda v: f(v, u_s), x_s)

def compute_B(x_s, u_s):
    return utilities.jacobian(lambda v: f(x_s, v), u_s)

unique_X = np.unique(X, axis=1)
unique_U = np.unique(U, axis=1)
# A = np.zeros((X.shape[0], X.shape[0]))
# B = np.zeros((X.shape[0], U.shape[0]))

# count = 0
# for i in range(unique_X.shape[1]):
#     for j in range(unique_U.shape[1]):
#         A += compute_A(unique_X[:, i], unique_U[:, j])
#         B += compute_B(unique_X[:, i], unique_U[:, j])
#         count += 1

# A /= count
# B /= count

# A = np.array([
#     [1, 0.02, 0,            0],
#     [0, 1,   -0.0134839336, 0.0000927891961],
#     [0, 0,    1,            0.02],
#     [0, 0,    0.312741646,  1]
# ])
A = np.array([
    [0.1, 0.02, 0,            0],
    [0,   0.1, -0.0134839336, 0.0000927891961],
    [0,   0,    0.1,          0.02],
    [0,   0,    0.312741646,  0.1]
])
B = np.array([
    [0.],
    [9750.11179],
    [0],
    [-14563.45848858]
])
# B = np.array([
#     [0],
#     [0.1],
#     [0],
#     [-0.1]
# ])

#%%
def lqr_dynamics(x, u):
    return A @ x + B @ u

total_datapoints = 0
while total_datapoints < num_datapoints:
    x = np.vstack(env.reset())
    done = False
    while not done and total_datapoints < num_datapoints:
        X[:, total_datapoints] = x[:, 0]
        u = np.array([[env.action_space.sample()]]) # Sampled from random agent
        U[:, total_datapoints] = u[:, 0]
        y = lqr_dynamics(x, u)
        Y[:, total_datapoints] = y[:, 0]
        x = y
        total_datapoints += 1

#%%

# num_episodes = 200
# num_steps_per_episode = 200
# x0 = np.array([
#     [-1],
#     [0],
#     [np.pi],
#     [0]
# ])
# action_range = np.arange(-1, 1, 0.1) # -1 to 1 in increments of 0.1
# for episode in range(num_episodes):
#     perturbation = np.array([
#         [0],
#         [0],
#         [np.random.normal(0, 0.05)],
#         [0]
#     ])
#     x = x0 + perturbation

#     for step in range(num_steps_per_episode):
#         X[:, episode+step] = x[:, 0]
#         u = np.random.choice(action_range) # random sample
#         U[:, episode+step] = u
#         # y = f(x, u)
#         y = odeint(f, x, tspan, args=(u))
#         Y[:, episode+step] = y[:, 0]

#         x = y

tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(2),
    regressor='ols'
)

#%% Training error
print("\nTesting error:")

num_episodes = 100
num_steps_per_episode = 200
testing_norms = np.zeros([X.shape[1]])

for episode in range(num_episodes):
    # perturbation = np.random.normal(0, 0.05)
    # x = np.array([
    #     [-1],
    #     [0],
    #     [np.pi + perturbation],
    #     [0]
    # ])
    x = np.vstack(env.reset())

    for step in range(num_steps_per_episode):
        phi_x = tensor.phi(x)
        u = np.array([[env.action_space.sample()]]) # Sampled from random agent

        predicted_x_prime = tensor.B.T @ tensor.K_(u) @ phi_x
        # true_x_prime = np.vstack(Y[:, i])
        true_x_prime = lqr_dynamics(x, u)

        testing_norms[(episode*num_steps_per_episode)+step] = utilities.l2_norm(true_x_prime, predicted_x_prime)

        x = true_x_prime

print("Mean testing norm:", np.mean(testing_norms))

#%% Define cost
def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vecs are snapshots
    mat = np.vstack(np.diag(x.T @ Q @ x)) + np.power(u, 2)*R
    return mat

# From Wen's homework (PA1)
# def c(x, u):
#     assert x.shape == (4,)
#     assert u.shape == (1,)
#     env = gym.make("env:CartPoleControlEnv-v0")
#     env.reset(state=x)
#     env.step(u)
#     observation, cost, done, info = env.step(u)
#     env.close()
#     return cost

# def reward_func(state, action):
#     #* assume state and action can be matrices where the columns are states/actions
#     x = state[0]
#     x_dot = state[1]
#     theta = state[2]
#     theta_dot = state[3]
#     # x, x_dot, theta, theta_dot = self.state
#     force = np.ones([state.shape[1]]) * -env.force_mag
#     force[np.where(action == 1)[0]] = env.force_mag
#     # force = env.force_mag if action == 1 else -env.force_mag
#     costheta = np.cos(theta)
#     sintheta = np.sin(theta)

#     # For the interested reader:
#     # https://coneural.org/florian/papers/05_cart_pole.pdf
#     temp = (
#         force + env.polemass_length * theta_dot ** 2 * sintheta
#     ) / env.total_mass
#     thetaacc = (env.gravity * sintheta - costheta * temp) / (
#         env.length * (4.0 / 3.0 - env.masspole * costheta ** 2 / env.total_mass)
#     )
#     xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass

#     if env.kinematics_integrator == "euler":
#         x = x + env.tau * x_dot
#         x_dot = x_dot + env.tau * xacc
#         theta = theta + env.tau * theta_dot
#         theta_dot = theta_dot + env.tau * thetaacc
#     else:  # semi-implicit euler
#         x_dot = x_dot + env.tau * xacc
#         x = x + env.tau * x_dot
#         theta_dot = theta_dot + env.tau * thetaacc
#         theta = theta + env.tau * theta_dot

#     # self.state = (x, x_dot, theta, theta_dot)

#     rewards = np.ones([state.shape[1]])
#     rewards[np.where(x < -env.x_threshold)] = 0.0
#     rewards[np.where(x > env.x_threshold)] = 0.0
#     rewards[np.where(theta < -env.theta_threshold_radians)] = 0.0
#     rewards[np.where(theta > env.theta_threshold_radians)] = 0.0

#     return rewards

# reward_func(np.array([[1, 1], [0, 0], [1, 1], [0, 0]]), np.array([[0, 0]]))

# def cost(x, u):
#     return -reward_func(x, u)

#%% Learn control
u_bounds = np.array([[0, 1]])
All_U = np.array([[-10, 10]])
gamma = 0.99
lamb = 1.0
lr = 1e-1
epsilon = 1e-3

algos = algorithmsv2.algos(
    X,
    All_U,
    u_bounds[0],
    tensor,
    cost,
    gamma=gamma,
    epsilon=epsilon,
    bellman_error_type=0,
    learning_rate=lr,
    weight_regularization_bool=True,
    weight_regularization_lambda=lamb,
    optimizer='adam'
)

# algos.w = np.load('bellman-weights.npy')
print("Weights before updating:", algos.w)
bellmanErrors, gradientNorms = algos.algorithm2(batch_size=512)
print("Weights after updating:", algos.w)

#%% Extract policy
All_U_range = np.arange(All_U.shape[1])
def policy(x):
    pis = algos.pis(x)[:,0]
    # Select action column at index sampled from policy distribution
    u_ind = np.random.choice(All_U_range, p=pis)
    u = np.vstack(All_U[:,u_ind])[0,0]
    u = 0 if u == -10 else 1
    return u

def random_policy():
    return env.action_space.sample()

#%% Test policy by simulating system
num_episodes = 200
rewards = np.zeros([num_episodes])
for episode in range(num_episodes):
    perturbation = np.random.normal(0, 0.05)
    x = np.array([
        [-1],
        [0],
        [np.pi + perturbation],
        [0]
    ])

    done = False
    while not done:
        # env.render()
        u = np.array([[policy(x)]])

        # next_state,reward,done,info = env.step(u)
        x = f(x, u)

        done = x[0, 0] < -env.x_threshold or \
               x[0, 0] > env.x_threshold or \
               x[2, 0] < -env.theta_threshold_radians or \
               x[2, 0] > env.theta_threshold_radians

        rewards[episode] += 1

print("Mean reward per episode:", np.mean(rewards))

#%%