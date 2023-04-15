# Imports
import matplotlib.pyplot as plt
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
from dynamics import all_actions, f

sys.path.append('../../../')
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy

#%% Load Koopman tensor with pickle
with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

# Variables
# gamma = 0.99
gamma = 1.0
reg_lambda = 1.0
# reg_lambda = 0.2

# Koopman value iteration policy
koopman_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    save_data_path="./analysis/tmp/discrete_value_iteration",
    seed=seed
)

# Train Koopman policy
# koopman_policy.train(training_epochs=2_000, batch_size=2**12)
koopman_policy.train(training_epochs=10, batch_size=2**14, how_often_to_chkpt=5)