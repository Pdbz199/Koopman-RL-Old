import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from cost import cost
from dynamics import (
    action_dim,
    all_actions,
    action_order,
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

# Koopman value iteration policy
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
    'saved_models/fluid-flow-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.003,
    load_model=True
)

x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
u = zero_policy(x)
soln = solve_ivp(fun=continuous_f(u), t_span=[0, 50.0], y0=x[:,0], method='RK45')
initial_state = soln.y[:,-1]
# initial_state = x

state = np.vstack(initial_state)
controlled_state = state

scale_factor = 150
num_steps = 19_000
alpha = np.empty((num_steps, state_dim))
for step in range(num_steps):
    alpha[step] = controlled_state[:,0] * scale_factor
    if step < 2_000:
        controlled_action = zero_policy(controlled_state)
    else:
        controlled_action, _ = koopman_policy.get_action(controlled_state)
    controlled_state = f(controlled_state, controlled_action)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(alpha[:,0], alpha[:,1], alpha[:,2])
ax.set_xlim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_ylim(-1.0 * scale_factor, 1.0 * scale_factor)
ax.set_zlim(0.0, 1.0 * scale_factor)
plt.show()

p = scipy.io.loadmat('../../../koopman_tensor/system_dynamics/fluid_flow/data/POD-MODES.mat')
Xavg = p['Xavg'] # (89351, 1)
Xdelta = p['Xdelta'] # (89351, 1)
Phi = p['Phi'] # (89351, 8)

snapshots = []
for k in range(0, alpha.shape[0], 100):
    u = Xavg[:,0] + Phi[:,0] * alpha[k,0] + Phi[:,1] * alpha[k,1] + Xdelta[:,0] * alpha[k,2]
    snapshots.append(u.reshape(449,199).T)

# Choose the FPS and the number of seconds to run for
fps = 8
num_seconds = 10

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8,4))
plt.axis("off")

a = snapshots[0]
im = plt.imshow(a, cmap='hot', clim=(-1,1))

def animate_func(i):
    # if i % fps == 0:
    #     print( '.', end ='' )

    im.set_array(snapshots[i])
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