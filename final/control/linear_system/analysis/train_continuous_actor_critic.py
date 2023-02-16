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
koopman_policy = System(
    true_dynamics=f,
    cost=cost,
    state_minimums=state_minimums,
    state_maximums=state_maximums,
    action_minimums=action_minimums,
    action_maximums=action_maximums,
    environment_steps=200,
    gradient_steps=1,
    init_steps=256,
    reward_scale=10,
    batch_size=256,
    is_episodic=True
)

# Train Koopman policy
koopman_policy.train_agent(num_training_iterations=2_000, initialization=False)
# koopman_policy.train_agent(num_training_iterations=2_000, initialization=True)

with open('./analysis/tmp/continuous_actor_critic/policy.pkl', 'wb') as file:
    pickle.dump(koopman_policy, file) # protocol=pickle.HIGHEST_PROTOCOL