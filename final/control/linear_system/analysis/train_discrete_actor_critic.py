# Imports
import matplotlib.pyplot as plt
import numpy as np
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
    action_dim,
    action_maximums,
    action_minimums,
    action_order,
    all_actions,
    f,
    state_dim,
    state_maximums,
    state_minimums,
    state_order
)

sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0

plot_path = f'output/discrete_actor_critic/seed_{seed}/'
plot_file_extensions = ['svg', 'png']

# Construct datasets
num_episodes = 100
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, size=[state_dim, N])
U = np.random.uniform(action_minimums, action_maximums, size=[action_dim, N])
Y = f(X, U)

# Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

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
    layer_1_dim=16,
    layer_2_dim=8,
    learning_rate=0.003
)
print(f"\nLearning rate: {koopman_policy.learning_rate}\n")

# Train Koopman policy
koopman_policy.train(num_training_episodes=5000, num_steps_per_episode=200)