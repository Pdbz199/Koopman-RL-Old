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
    dt,
    f,
    state_maximums,
    state_minimums
)

sys.path.append('../../../')
from final.control.policies.traditional_discrete_actor_critic import DiscretePolicyIterationPolicy

#%% Load Koopman tensor with pickle
with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

# Variables
# gamma = 0.99
gamma = 1.0
reg_lambda = 1.0

# Neural network value iteration policy
koopman_policy = DiscretePolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    save_data_path="./analysis/tmp/traditional_discrete_actor_critic",
    seed=seed,
    learning_rate=0.003,
    layer_1_dim=128,
    layer_2_dim=256
)
print(f"\nLearning rate: {koopman_policy.learning_rate}\n")

# Train Koopman policy
koopman_policy.train(num_training_episodes=10_000, num_steps_per_episode=int(20 / dt), how_often_to_chkpt=200)