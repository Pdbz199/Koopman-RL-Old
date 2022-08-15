import matplotlib.pyplot as plt
import numpy as np

seed = 123
np.random.seed(123)

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
B = np.array([

])

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

        B @ B * u

        return [ x_dot, y_dot ]

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')

    return np.vstack(soln.y[:,-1])

#%% Generate data
num_episodes = 200
num_steps_per_episode = int(20.0 / dt)
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

#%% Plot one of the trajectories
plt.title("Dynamics over time (trajectory 1 from dataset)")
plt.xlabel("x")
plt.ylabel("step")
plt.plot(np.arange(num_steps_per_episode), X[0,:num_steps_per_episode])
plt.plot(np.arange(num_steps_per_episode), X[1,:num_steps_per_episode])
plt.legend(labels=['x_1', 'x_2'])
plt.show()

#%% Plot x1 v x2
plt.title("x_1 vs. x_2 (trajectory 1 from dataset)")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.plot(X[0,:num_steps_per_episode], X[1,:num_steps_per_episode])
plt.show()

#%% Generate trajectory from Koopman tensor
true_xs = np.zeros([state_dim,num_steps_per_episode])
koopman_xs = np.zeros([state_dim,num_steps_per_episode])

x = np.random.rand(*state_column_shape) * 3 * np.random.choice([-1,1], size=state_column_shape)
true_x = x
koopman_x = x
for step in range(num_steps_per_episode):
    true_xs[:,step] = true_x[:,0]
    koopman_xs[:,step] = koopman_x[:,0]
    u = zero_policy(x)
    # u = random_policy(x)
    true_x = f(true_x, u)
    koopman_x = tensor.f(koopman_x, u)

#%% Plot comparison trajectories (true vs. Koopman)
fig, axs = plt.subplots(2)
fig.suptitle("Comparison Trajectories (True vs. Koopman)")

axs[0].set_title("Dynamics Over Time (True)")
axs[0].set(xlabel="step", ylabel="x")
axs[0].plot(np.arange(num_steps_per_episode), true_xs[0])
axs[0].plot(np.arange(num_steps_per_episode), true_xs[1])

axs[1].set_title("Dynamics Over Time (Koopman)")
axs[1].set(xlabel="step", ylabel="x")
axs[1].plot(np.arange(num_steps_per_episode), koopman_xs[0])
axs[1].plot(np.arange(num_steps_per_episode), koopman_xs[1])

fig.legend(labels=['x_1', 'x_2'])

plt.tight_layout()
plt.show()

#%% Plot x1 v x2
fig, axs = plt.subplots(2)
fig.suptitle("x_1 vs. x_2 (True vs. Koopman)")

axs[1].set_title("x_1 vs. x_2 (True)")
axs[1].set(xlabel="x_1", ylabel="x_2")
axs[1].plot(true_xs[0], true_xs[1])

axs[0].set_title("x_1 vs. x_2 (Koopman)")
axs[0].set(xlabel="x_1", ylabel="x_2")
axs[0].plot(koopman_xs[0], koopman_xs[1])

plt.tight_layout()
plt.show()

#%% Plot differences
fig, axs = plt.subplots(2)
fig.suptitle("Distribution of Errors")

axs[0].set_title("True state - Koopman state")
axs[0].set(xlabel="Error", ylabel="Count")
axs[0].hist(np.sum(true_xs - koopman_xs, axis=0))

axs[1].set_title("(True state - Koopman state)^2")
axs[1].set(xlabel="Error", ylabel="Count")
axs[1].hist(np.sum((true_xs - koopman_xs)**2, axis=0))

plt.tight_layout()
plt.show()