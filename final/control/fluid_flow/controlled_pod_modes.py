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
    'saved_models/fluid-flow-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.003,
    load_model=True
)

# Koopman value iteration policy
# koopman_policy = DiscreteKoopmanValueIterationPolicy(
#     f,
#     gamma,
#     reg_lambda,
#     tensor,
#     all_actions,
#     cost,
#     'saved_models/fluid-flow-discrete-value-iteration-policy.pt',
#     dt=dt,
#     seed=seed
# )

x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
u = zero_policy(x)
soln = solve_ivp(fun=continuous_f(u), t_span=[0, 50.0], y0=x[:,0], method='RK45')
initial_state = soln.y[:,-1]

# initial_state = x

# initial_states = np.random.uniform(
#     state_minimums,
#     state_maximums,
#     [state_dim, 1]
# ).T
# initial_state = initial_states[0]

state = np.vstack(initial_state)
koopman_state = state
lqr_state = state

scale_factor = 150
num_steps = int(10_000 / dt)
koopman_alpha = np.empty((num_steps, state_dim))
koopman_costs = np.empty((num_steps,1))
lqr_alpha = np.empty((num_steps, state_dim))
lqr_costs = np.empty((num_steps,1))
for step in range(num_steps):
    koopman_alpha[step] = koopman_state[:,0] * scale_factor
    lqr_alpha[step] = lqr_state[:,0] * scale_factor

    if step < 2_000:
        koopman_action = zero_policy(koopman_state)
        lqr_action = zero_policy(lqr_state)
    else:
        try:
            koopman_action, _ = koopman_policy.get_action(koopman_state)
        except:
            koopman_action = koopman_policy.get_action(koopman_state)
        lqr_action = lqr_policy.get_action(lqr_state)

    lqr_costs[step] = cost(lqr_state, lqr_action)
    koopman_costs[step] = cost(koopman_state, koopman_action)

    koopman_state = f(koopman_state, koopman_action)
    lqr_state = f(lqr_state, lqr_action)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(lqr_alpha[:,0], lqr_alpha[:,1], lqr_alpha[:,2])
ax.plot3D(koopman_alpha[:,0], koopman_alpha[:,1], koopman_alpha[:,2])
ax.set_xlim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_ylim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_zlim(0.0, 1.0 * scale_factor)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(321, projection='3d')
ax.plot3D(lqr_alpha[:,0], lqr_alpha[:,1], lqr_alpha[:,2])
ax.set_xlim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_ylim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_zlim(0.0, 1.0 * scale_factor)
ax = fig.add_subplot(322, projection='3d')
ax.plot3D(koopman_alpha[:,0], koopman_alpha[:,1], koopman_alpha[:,2])
ax.set_xlim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_ylim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_zlim(0.0, 1.0 * scale_factor)
ax = fig.add_subplot(323)
ax.plot(lqr_alpha)
ax = fig.add_subplot(324)
ax.plot(koopman_alpha)
ax = fig.add_subplot(325)
ax.plot(lqr_costs)
ax = fig.add_subplot(326)
ax.plot(koopman_costs)
plt.show()

print(f"Average LQR cost: {lqr_costs.mean()}")
print(f"Average Koopman cost: {koopman_costs.mean()}")

p = scipy.io.loadmat('../../../koopman_tensor/system_dynamics/fluid_flow/data/POD-MODES.mat')
Xavg = p['Xavg'] # (89351, 1)
Xdelta = p['Xdelta'] # (89351, 1)
Phi = p['Phi'] # (89351, 8)

koopman_snapshots = []
lqr_snapshots = []
for k in range(0, koopman_alpha.shape[0], 100):
    u = Xavg[:,0] + Phi[:,0] * koopman_alpha[k,0] + Phi[:,1] * koopman_alpha[k,1] + Xdelta[:,0] * koopman_alpha[k,2]
    koopman_snapshots.append(u.reshape(449,199).T)

    u = Xavg[:,0] + Phi[:,0] * lqr_alpha[k,0] + Phi[:,1] * lqr_alpha[k,1] + Xdelta[:,0] * lqr_alpha[k,2]
    lqr_snapshots.append(u.reshape(449,199).T)

# Choose the FPS and the number of seconds to run for
fps = 8
num_seconds = 10

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8,4))
plt.axis("off")

a = lqr_snapshots[0]
im = plt.imshow(a, cmap='hot', clim=(-1,1))

def animate_func(i):
    # if i % fps == 0:
    #     print( '.', end ='' )

    im.set_array(lqr_snapshots[i])
    # plt.imsave(f'output/pod_modes/frame_{i}.png', snapshots[i], cmap='hot', vmin=-1, vmax=1)
    return [im]

anim = FuncAnimation(
    fig,
    animate_func,
    frames = num_seconds * fps,
    interval = 1000 / fps # in ms
)

plt.show()
# FFwriter = FFMpegWriter(fps=fps)
# anim.save('stabilizing_fluid_flow.mp4', writer=FFwriter)