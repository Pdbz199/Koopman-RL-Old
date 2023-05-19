# Imports
import numpy as np
import pickle
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
    all_actions,
    f,
    state_maximums,
    state_minimums
)

sys.path.append('../../../')
from final.control.policies.proximal_policy_optimization import ProximalPolicyOptimization

#%% Load Koopman tensor with pickle
with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

# Variables
gamma = 0.99
# gamma = 1.0
# gamma = 0.33
reg_lambda = 1.0

# Neural network value iteration policy
koopman_policy = ProximalPolicyOptimization(
    env=f,
    all_actions=all_actions,
    is_continuous=False,
    dynamics_model=tensor,
    state_minimums=state_minimums,
    state_maximums=state_maximums,
    cost=cost,
    save_data_path="./analysis/tmp/proximal_policy_optimization",
    gamma=gamma,
    value_beta=1.0,
    entropy_beta=0.01,
    learning_rate=3e-4,
    is_gym_env=False,
    seed=seed
)
print(f"\nLearning rate: {koopman_policy.learning_rate}\n")

# Train Koopman policy
num_episodes = 100_000
ppo_steps = 5
ppo_clip = 0.2
reward_threshold = 200
print_every = 250
num_trials = print_every

koopman_policy.train(
    num_episodes,
    num_trials,
    ppo_steps,
    ppo_clip,
    reward_threshold,
    print_every,
    num_steps_per_episode=200
)