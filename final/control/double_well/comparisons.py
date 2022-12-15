import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from cost import cost, Q, R, reference_point
from dynamics import (
    action_dim,
    all_actions,
    action_order,
    continuous_A,
    continuous_B,
    continuous_f,
    dt,
    f,
    state_column_shape,
    state_dim,
    state_minimums,
    state_maximums,
    state_order,
    zero_policy
)
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.integrate import solve_ivp

sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from final.control.policies.lqr import LQRPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0
# reg_lambda = 0.1

# LQR Policy
lqr_policy = LQRPolicy(
    continuous_A,
    continuous_B,
    Q,
    R,
    reference_point,
    gamma,
    reg_lambda,
    dt=dt,
    is_continuous=True,
    seed=seed
)

# Dummy Koopman tensor
N = 10
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

# Koopman discrete actor critic policy
koopman_policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'saved_models/lorenz-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.003,
    load_model=True
)

# Koopman value iteration policy
koopman_policy_2 = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    'saved_models/lorenz-discrete-value-iteration-policy.pt',
    dt=dt,
    seed=seed
)

# x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
# u = zero_policy(x)
# soln = solve_ivp(fun=continuous_f(u), t_span=[0, 50.0], y0=x[:,0], method='RK45')
# initial_state = soln.y[:,-1]

# initial_state = x

initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, 1]
).T
initial_state = initial_states[0]

state = np.vstack(initial_state)
koopman_state = state
koopman_state_2 = state
lqr_state = state

num_steps = int(100.0 / dt)
koopman_alpha = np.empty((num_steps, state_dim))
koopman_alpha_2 = np.empty((num_steps, state_dim))
koopman_costs = np.empty((num_steps,1))
koopman_costs_2 = np.empty((num_steps,1))
lqr_alpha = np.empty((num_steps, state_dim))
lqr_costs = np.empty((num_steps,1))
for step in range(num_steps):
    koopman_alpha[step] = koopman_state[:,0]
    koopman_alpha_2[step] = koopman_state_2[:,0]
    lqr_alpha[step] = lqr_state[:,0]

    if step < 20:
        koopman_action = zero_policy(koopman_state)
        koopman_action_2 = zero_policy(koopman_state_2)
        lqr_action = zero_policy(lqr_state)
    else:
        koopman_action, _ = koopman_policy.get_action(koopman_state)
        koopman_action_2 = koopman_policy_2.get_action(koopman_state_2)
        lqr_action = lqr_policy.get_action(lqr_state)

    lqr_costs[step] = cost(lqr_state, lqr_action)
    koopman_costs[step] = cost(koopman_state, koopman_action)
    koopman_costs_2[step] = cost(koopman_state_2, koopman_action_2)

    koopman_state = f(koopman_state, koopman_action)
    koopman_state_2 = f(koopman_state_2, koopman_action_2)
    lqr_state = f(lqr_state, lqr_action)

label_names = [
    "LQR",
    "Koopman AC",
    "Koopman VI"
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(lqr_alpha[:,0], lqr_alpha[:,1], lqr_alpha[:,2])
ax.plot3D(koopman_alpha[:,0], koopman_alpha[:,1], koopman_alpha[:,2])
ax.plot3D(koopman_alpha_2[:,0], koopman_alpha_2[:,1], koopman_alpha_2[:,2])
ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
plt.legend(label_names)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(331, projection='3d')
ax.plot3D(lqr_alpha[:,0], lqr_alpha[:,1], lqr_alpha[:,2])
ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
ax = fig.add_subplot(332, projection='3d')
ax.plot3D(koopman_alpha[:,0], koopman_alpha[:,1], koopman_alpha[:,2])
ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
ax = fig.add_subplot(333, projection='3d')
ax.plot3D(koopman_alpha_2[:,0], koopman_alpha_2[:,1], koopman_alpha_2[:,2])
ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
ax = fig.add_subplot(334)
ax.plot(lqr_alpha)
ax = fig.add_subplot(335)
ax.plot(koopman_alpha)
ax = fig.add_subplot(336)
ax.plot(koopman_alpha_2)
ax = fig.add_subplot(337)
ax.plot(lqr_costs)
ax = fig.add_subplot(338)
ax.plot(koopman_costs)
ax = fig.add_subplot(339)
ax.plot(koopman_costs_2)
plt.show()

print(f"Average LQR cost: {lqr_costs.mean()}")
print(f"Average Koopman AC cost: {koopman_costs.mean()}")
print(f"Average Koopman VI cost: {koopman_costs_2.mean()}")

# p = scipy.io.loadmat('../../../koopman_tensor/system_dynamics/fluid_flow/data/POD-MODES.mat')
# Xavg = p['Xavg'] # (89351, 1)
# Xdelta = p['Xdelta'] # (89351, 1)
# Phi = p['Phi'] # (89351, 8)

# koopman_snapshots = []
# lqr_snapshots = []
# for k in range(0, koopman_alpha.shape[0], 100):
#     u = Xavg[:,0] + Phi[:,0] * koopman_alpha[k,0] + Phi[:,1] * koopman_alpha[k,1] + Xdelta[:,0] * koopman_alpha[k,2]
#     koopman_snapshots.append(u.reshape(449,199).T)

#     u = Xavg[:,0] + Phi[:,0] * lqr_alpha[k,0] + Phi[:,1] * lqr_alpha[k,1] + Xdelta[:,0] * lqr_alpha[k,2]
#     lqr_snapshots.append(u.reshape(449,199).T)

# Choose the FPS and the number of seconds to run for
# fps = 8
# num_seconds = 10

# First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure(figsize=(8,4))
# plt.axis("off")

# a = lqr_snapshots[0]
# a = koopman_snapshots[0]
# im = plt.imshow(a, cmap='hot', clim=(-1,1))

# def animate_func(i):
#     # if i % fps == 0:
#     #     print( '.', end ='' )

#     im.set_array(koopman_snapshots[i])
#     # plt.imsave(f'output/pod_modes/frame_{i}.png', koopman_snapshots[i], cmap='hot', vmin=-1, vmax=1)
#     return [im]

# anim = FuncAnimation(
#     fig,
#     animate_func,
#     frames = num_seconds * fps,
#     interval = 1000 / fps # in ms
# )

# plt.show()
# FFwriter = FFMpegWriter(fps=fps)
# anim.save('stabilizing_fluid_flow.mp4', writer=FFwriter)