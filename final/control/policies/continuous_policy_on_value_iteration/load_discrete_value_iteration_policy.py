#%% Imports
import sys

sys.path.append('../../../../')
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy

def load_koopman_value_iteration_policy(
    system_dynamics,
    tensor,
    all_actions,
    cost,
    seed,
    gamma=0.99,
    regularization_lambda=1.0,
    system_name="linear_system"
):
    return DiscreteKoopmanValueIterationPolicy(
        system_dynamics,
        gamma,
        regularization_lambda,
        tensor,
        all_actions,
        cost,
        save_data_path=f"../../{system_name}/analysis/tmp/discrete_value_iteration",
        seed=seed,
        load_model=True
    )