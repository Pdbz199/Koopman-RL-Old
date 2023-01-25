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
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy

#%% Load Koopman tensor with pickle
with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

# Variables
gamma = 0.99
reg_lambda = 1.0

# Koopman value iteration policy
koopman_policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    save_data_path="./analysis/tmp/discrete_actor_critic",
    seed=seed,
    dt=dt,
    learning_rate=0.003
)
print(f"\nLearning rate: {koopman_policy.learning_rate}\n")

# Train Koopman policy
koopman_policy.train(num_training_episodes=2_000, num_steps_per_episode=int(20 / dt))