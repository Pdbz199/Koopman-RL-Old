import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from control import dlqr

sys.path.append('./')
from dynamics import (
    f,
    state_dim,
    state_maximums,
    state_minimums,
    zero_policy
)
from cost import Q, R

# Add to path to find final module
sys.path.append('../../../')

with open('./analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    koopman_tensor = pickle.load(handle)

if __name__ == "__main__":
    phi_dim = koopman_tensor.phi_dim

    dummy_state = np.vstack([0, 0, 0])
    K_0 = koopman_tensor.K_(zero_policy(dummy_state))
    print("K^0 shape:", K_0.shape)
    U, sigma, V_T = np.linalg.svd(K_0)
    print("U, Σ, and V shapes:", U.shape, sigma.shape, V_T.shape)

    left_singular_function_1 = np.vstack(U[0])
    left_singular_function_2 = np.vstack(U[1])
    left_singular_function_3 = np.vstack(U[2])
    print("Left singular function 1's shape:", left_singular_function_1.shape)

    padded_B = np.zeros((20, 1))
    padded_B[1] = 1
    padded_Q = np.pad(Q, ((0, 17), (0, 17)))
    print("Padded B shape:", padded_B.shape)
    print("Padded Q shape:", padded_Q.shape)
    print("R:", R)

    C, P, E = dlqr(K_0, padded_B, padded_Q, R)
    print("C shape:", C.shape)
    print("P shape:", P.shape)
    print("E shape:", E.shape)

    diagonal_control = np.zeros((1, state_dim))
    diagonal_control[0, 0] = C @ left_singular_function_1
    diagonal_control[0, 1] = C @ left_singular_function_2
    diagonal_control[0, 2] = C @ left_singular_function_3
    print("Diagonal control:\n", diagonal_control)

    def optimal_policy(x):
        return -diagonal_control @ x

    num_steps = 10000

    xs = np.zeros((num_steps, state_dim))
    x = np.vstack([0, 0, 1])

    for step_num in range(num_steps):
        xs[step_num] = x[:, 0]
        u = optimal_policy(x)
        x = f(x, u)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_xlim(state_minimums[0, 0], state_maximums[0, 0])
        ax.set_ylim(state_minimums[1, 0], state_maximums[1, 0])
        ax.set_zlim(state_minimums[2, 0], state_maximums[2, 0])
        ax.plot(xs[:frame, 0], xs[:frame, 1], xs[:frame, 2])

        if frame == num_steps-1: print("DONE")

    anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=50, repeat=False)
    plt.show()