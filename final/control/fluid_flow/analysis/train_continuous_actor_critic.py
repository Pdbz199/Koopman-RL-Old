# Imports
import gym
import numpy as np
import pickle
# import pybullet_envs
import sys

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

sys.path.append('./')
from cost import reward
from dynamics import (
    action_maximums,
    action_minimums,
    all_actions,
    dt,
    f,
    state_dim,
    state_maximums,
    state_minimums
)

sys.path.append('../../../')
from final.control.policies.koopman_soft_actor_critic.agent import System

#%% Load Koopman tensor with pickle
with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    koopman_model = pickle.load(handle)

# Variables
gamma = 0.99
reg_lambda = 1.0

# steps_per_iteration = 200
# steps_per_iteration = int(2 / dt)
steps_per_iteration = int(20 / dt)

# Koopman soft actor-critic policy
koopman_policy = System(
    is_gym_env=False,
    true_dynamics=f,
    koopman_model=koopman_model,
    reward=reward,
    state_minimums=state_minimums,
    state_maximums=state_maximums,
    action_minimums=action_minimums,
    action_maximums=action_maximums,
    environment_steps=steps_per_iteration,
    gradient_steps=1,
    init_steps=steps_per_iteration,
    reward_scale=5,
    batch_size=2**10,
    is_episodic=True,
    # learning_rate=3e-4
)

# Train Koopman policy
num_training_iterations = 2_000
# num_training_iterations = 20_000
# initialization = False
initialization = True
koopman_policy.train_agent(num_training_iterations, initialization)

# Save koopman policy
with open('./analysis/tmp/continuous_actor_critic/policy.pkl', 'wb') as file:
    pickle.dump(koopman_policy, file) # protocol=pickle.HIGHEST_PROTOCOL