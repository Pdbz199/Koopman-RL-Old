#%% Imports
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
from cost import Q, R, reference_point
from dynamics import continuous_A, continuous_B

sys.path.append('../../../')
from final.control.policies.lqr import LQRPolicy

#%% Variables
gamma = 0.99
reg_lambda = 1.0

path = './analysis/tmp'

#%% LQR policy
lqr_policy = LQRPolicy(
    continuous_A,
    continuous_B,
    Q,
    R,
    reference_point,
    gamma,
    reg_lambda,
    seed=seed,
    is_continuous=True
)

#%% Save LQR policy
with open(f'{path}/lqr/policy.pickle', 'wb') as handle:
    pickle.dump(lqr_policy, handle)
    print("Saved LQR policy")