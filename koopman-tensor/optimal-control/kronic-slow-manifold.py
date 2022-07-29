import matplotlib.pyplot as plt
import numpy as np

seed = 123
np.random.seed(123)

from control import lqr
from scipy.integrate import solve_ivp
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables

#%% Dynamics variables
state_dim = 2
action_dim = 1

state_order = 2
action_order = 2

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

phi_column_shape = [phi_dim, 1]

action_range = np.array([-5, 5])
step_size = 1.0
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

mu = -0.1
lamb = -1
B_1 = np.array([
    [0],
    [1]
])
# B_1 = np.array([
#     [1],
#     [1]
# ])
# B_1 = np.array([
#     [1],
#     [0]
# ])

dt = 0.01
t_span = np.arange(0, dt, dt/10)

#%% Default policy functions
def zero_policy(x):
    return np.zeros(action_column_shape)

def random_policy(x):
    return np.random.choice(all_actions, size=action_column_shape)

#%% Continuous-time function
def continuous_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - state vector
            t - timestep
        """
        x, y = input

        x_dot = mu * x
        y_dot = lamb * ( y - x**2 )

        u = action
        if u is None:
            u = random_policy(x_dot)

        # control = B_1 * u

        return [ x_dot + u[0,0], y_dot + u[0,0] ]

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    # u = action[:,0]

    soln = solve_ivp(fun=continuous_f(action), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')

    return np.vstack(soln.y[:,-1])

continuous_A = np.array([
    [mu, 0, 0],
    [0, lamb, -lamb],
    [0, 0, 2*mu]
])
continuous_B = np.array([
    [0],
    [1],
    [0]
])

#%% Reward/Cost
Q_ = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
R = 1

#%% LQR using KOOC/KRONIC
C = lqr(continuous_A, continuous_B, Q_, R)[0]

def lqr_policy(x):
    return -C @ x

#%% Generate data
num_episodes = 200
num_steps_per_episode = int(30.0 / dt)
N = num_episodes*num_steps_per_episode # Number of datapoints
X = np.zeros([state_dim,N])
Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

for episode in range(num_episodes):
    x = np.random.rand(*state_column_shape) * 3 * np.random.choice([-1,1], size=state_column_shape)
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = x[:,0]
        # u = zero_policy(x)
        u = random_policy(x)
        U[:,(episode*num_steps_per_episode)+step] = u[:,0]
        x = f(x, u)
        Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

#%%
kronic_xs = np.zeros([num_steps_per_episode,state_dim])
koopman_xs = np.zeros([num_steps_per_episode,state_dim])

x = np.random.rand(*state_column_shape) * 3 * np.random.choice([-1,1], size=state_column_shape)
kronic_x = x
koopman_x = x
for step in range(num_steps_per_episode):
    kronic_xs[step] = kronic_x[:,0]
    koopman_xs[step] = koopman_x[:,0]

    phi_kronic_x = np.append(kronic_x, [[kronic_x[0,0]**2]], axis=0)
    phi_koopman_x = np.append(koopman_x, [[koopman_x[0,0]**2]], axis=0)

    kronic_u = lqr_policy(phi_kronic_x)
    koopman_u = zero_policy(phi_koopman_x)

    kronic_x = f(kronic_x, kronic_u)
    koopman_x = f(koopman_x, koopman_u)

data_points_range = np.arange(num_steps_per_episode)

fig, axs = plt.subplots(2)
fig.suptitle("Dynamics Over Time (KRONIC vs. Koopman)")

axs[0].set_title("KRONIC dynamics")
axs[0].set(xlabel="Step", ylabel="State value")

axs[1].set_title("Koopman dynamics")
axs[1].set(xlabel="Step", ylabel="State value")

for i in range(state_dim):
    axs[0].plot(data_points_range, kronic_xs[:,i])
    axs[1].plot(data_points_range, koopman_xs[:,i])

# fig.legend(labels=['x_1', 'x_2'])

plt.tight_layout()
plt.show()