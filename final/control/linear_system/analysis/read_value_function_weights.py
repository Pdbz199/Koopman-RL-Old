import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import torch

sys.path.append("./")
from dynamics import f, get_random_initial_conditions

sys.path.append("../../../")
from observables import monomials

if __name__ == '__main__':
    #%% Load LQR policy
    with open('./analysis/tmp/lqr/policy.pickle', 'rb') as handle:
        lqr_policy = pickle.load(handle)

    A = lqr_policy.A
    B = lqr_policy.B
    Q = lqr_policy.Q
    R = lqr_policy.R

    p = B.shape[1]

    X = lqr_policy.lqr_soln[1]
    print(f"Solution to Riccati Equation:\n{X}\n")

    Q_t_plus_1 = X
    c_t_plus_1 = 0
    # c_t_plus_1 = -8898.5 * 2

    cross_terms = Q + A.T @ Q_t_plus_1 @ A - \
                    A.T @ Q_t_plus_1 @ B @ \
                    np.linalg.inv(R + B.T @ Q_t_plus_1 @ B) @ \
                    B.T @ Q_t_plus_1 @ A
    print(f"Cross terms:\n{cross_terms / 2}\n")
    all_coeffs = np.zeros(6)
    all_coeffs[0] = cross_terms[0, 0] / 2
    all_coeffs[1] = cross_terms[0, 1]
    all_coeffs[2] = cross_terms[0, 2]
    all_coeffs[3] = cross_terms[1, 1] / 2
    all_coeffs[4] = cross_terms[1, 2]
    all_coeffs[5] = cross_terms[2, 2] / 2
    print(f"Squared term weights:\n{np.vstack(all_coeffs)}\n")

    def V_lqr(x):
        return (1/2) * (
            x.T @ ( cross_terms ) @ x + \
            (p/2) + (1/2) * (np.log(np.linalg.det(R + B.T @ Q_t_plus_1 @ B))) - \
            (p/2) * np.log(2 * np.pi * np.exp(1)) + \
            (1/2) * np.trace(Q_t_plus_1) + c_t_plus_1
        )

    value_function_weights = torch.load('./analysis/tmp/discrete_value_iteration/policy.pt')
    print(f"Value Function Weights:\n{value_function_weights}\n")

    print(f"Norm of difference between value function weights and LQR weights:\n{np.linalg.norm(all_coeffs - value_function_weights.numpy()[4:])}\n")

    phi = monomials(2)
    # value_function_weights[0, 0] = 0 # coefficient on constant
    def V_value_iteration(x):
        return value_function_weights.T @ phi(x)

    num_steps = 100

    lqr_vs = np.zeros(num_steps)
    value_iteration_vs = np.zeros(num_steps)

    # initial_state = np.vstack([5, 1, 3])
    # state = initial_state
    initial_states = get_random_initial_conditions(num_samples=num_steps)
    for step_num in range(num_steps):
        state = np.vstack(initial_states[step_num])
        lqr_vs[step_num] = V_lqr(state)
        value_iteration_vs[step_num] = V_value_iteration(state)
        # action = lqr_policy.get_action(state)
        # state = f(state, action)
    plt.title("V*(x) vs V(x)")
    plt.xlabel("Sample number")
    plt.ylabel("Value")
    plt.plot(lqr_vs)
    plt.plot(value_iteration_vs)
    plt.show()