# Imports
import gym
import numpy as np
import pickle
import pybullet_envs
import sys

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from cost import cost
from dynamics import (
    action_maximums,
    action_minimums,
    all_actions,
    f,
    state_dim,
    state_maximums,
    state_minimums
)

sys.path.append('../../../')
from final.control.policies.soft_actor_critic.agent import System

#%% Load Koopman tensor with pickle
# with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
#     tensor = pickle.load(handle)

# Variables
gamma = 0.99
reg_lambda = 1.0

# Koopman soft actor-critic policy
# koopman_policy = System(
#     true_dynamics=f,
#     cost=cost,
#     state_minimums=state_minimums,
#     state_maximums=state_maximums,
#     action_minimums=action_minimums,
#     action_maximums=action_maximums,
#     environment_steps=200,
#     gradient_steps=1,
#     init_steps=256,
#     reward_scale=10,
#     batch_size=256,
#     is_episodic=True
# )

# koopman_policy = System(
#     is_gym_env=True,
#     true_dynamics=gym.make('CartPole-v1'),
#     cost=cost,
#     state_minimums=np.array([[-4.8],[-np.inf],[-0.42],[-np.inf]]),
#     state_maximums=np.array([[4.8],[np.inf],[0.42],[np.inf]]),
#     action_minimums=np.array([[0.0]]),
#     action_maximums=np.array([[1.0]]),
#     environment_steps=200,
#     gradient_steps=1,
#     init_steps=256,
#     reward_scale=10,
#     batch_size=256,
#     is_episodic=True,
#     render_env=True
# )

# koopman_policy = System(
#     is_gym_env=True,
#     true_dynamics=gym.make('BipedalWalker-v3'),
#     cost=cost,
#     state_minimums=np.vstack([3.14, 5.0, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
#     state_maximums=-np.vstack([3.14, 5.0, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
#     action_minimums=np.vstack([-1.0, -1.0, -1.0, -1.0]),
#     action_maximums=np.vstack([1.0, 1.0, 1.0, 1.0]),
#     environment_steps=10_000,
#     memory_capacity=1_000_000,
#     gradient_steps=1,
#     init_steps=10_000,
#     reward_scale=4,
#     batch_size=256,
#     is_episodic=True,
#     render_env=False,
#     tau=1e-3,
#     learning_rate=3e-4
# )

koopman_policy = System(
    is_gym_env=True,
    true_dynamics=gym.make('InvertedPendulumBulletEnv-v0'),
    cost=cost,
    state_minimums=np.vstack([0.0 for _ in range(5)]),
    state_maximums=-np.vstack([0.0 for _ in range(5)]),
    action_minimums=np.vstack([-1.0, -1.0, -1.0, -1.0]),
    action_maximums=np.vstack([1.0, 1.0, 1.0, 1.0]),
    environment_steps=10_000,
    memory_capacity=1_000_000,
    gradient_steps=1,
    init_steps=10_000,
    reward_scale=2,
    batch_size=256,
    is_episodic=True,
    render_env=False,
    tau=5e-3,
    learning_rate=3e-4
)

# Train Koopman policy
num_training_iterations = 400
initialization = True
koopman_policy.train_agent(num_training_iterations, initialization)

# Save koopman policy
with open('./analysis/tmp/continuous_actor_critic/policy.pkl', 'wb') as file:
    pickle.dump(koopman_policy, file) # protocol=pickle.HIGHEST_PROTOCOL