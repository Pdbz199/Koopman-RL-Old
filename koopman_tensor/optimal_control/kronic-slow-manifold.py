import matplotlib.pyplot as plt
import numpy as np
import torch

seed = 123
np.random.seed(123)

from control import lqr, care
from scipy.integrate import solve_ivp
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor
sys.path.append('../../')
import observables

PATH = './manifold-value-model.pt'

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
lamb = 1
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
        x_1, x_2 = input

        u = action
        if u is None:
            u = random_policy(x_1)

        x_1_dot = mu * x_1
        x_2_dot = lamb * ( x_2 - x_1**2 )

        control = B_1 * u

        return [ x_1_dot + control[0,0], x_2_dot + control[1,0] ]

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
Q_ = np.eye(state_dim)
lifted_Q_ = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
R = 1

w_r = np.array([
    [0],
    [0]
])

lifted_w_r = np.array([
    [0],
    [0],
    [0]
])

def cost(x, u):
    # Assuming that data matrices are passed in for X and U. Columns vectors are snapshots
    _x = x - w_r
    mat = np.vstack(np.diag(_x.T @ Q_ @ _x)) + np.power(u, 2)*R
    # x.T @ Q @ x + u.T @ R @ u
    return mat.T

def lifted_cost(x, u):
    _x = x - lifted_w_r
    mat = np.vstack(np.diag(_x.T @ lifted_Q_ @ _x)) + np.power(u, 2)*R
    # x.T @ Q @ x + u.T @ R @ u
    return mat.T 

#%% LQR using KOOC/KRONIC
default_C = lqr(continuous_A, continuous_B, lifted_Q_, R)[0]

gamma = 0.99
lqr_lamb = 1e-5 # Should be 1.0 I think

P = care(continuous_A*np.sqrt(gamma), continuous_B*np.sqrt(gamma), lifted_Q_, R)[0]
entropy_C = np.linalg.inv(R + gamma*continuous_B.T @ P @ continuous_B) @ (gamma*continuous_B.T @ P @ continuous_A)
sigma_t = lqr_lamb * np.linalg.inv(R + continuous_B.T @ P @ continuous_B)

def lqr_policy(x, entropy=True):
    _x = x - lifted_w_r

    if entropy:
        return np.random.normal(-entropy_C @ _x, sigma_t)

    return -default_C @ _x

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

#%% Policy function as PyTorch model
def inner_pi_us(us, xs):
    phi_x_primes = tensor.K_(us) @ tensor.phi(xs) # us.shape[1] x dim_phi x xs.shape[1]

    V_x_primes_arr = torch.zeros([all_actions.shape[0], xs.shape[1]])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T # (1, xs.shape[1])

    inner_pi_us_values = -(torch.from_numpy(cost(xs, us)).float() + gamma * V_x_primes_arr) # us.shape[1] x xs.shape[1]

    return inner_pi_us_values * (1 / lamb) # us.shape[1] x xs.shape[1]

def pis(xs):
    delta = np.finfo(np.float32).eps # 1e-25

    inner_pi_us_response = torch.real(inner_pi_us(np.array([all_actions]), xs)) # all_actions.shape[0] x xs.shape[1]

    # Max trick
    max_inner_pi_u = torch.amax(inner_pi_us_response, axis=0) # xs.shape[1]
    diff = inner_pi_us_response - max_inner_pi_u

    pi_us = torch.exp(diff) + delta # all_actions.shape[0] x xs.shape[1]
    Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]
    
    return pi_us / Z_x # all_actions.shape[0] x xs.shape[1]

def discrete_bellman_error(batch_size):
    ''' Equation 12 in writeup '''
    x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
    x_batch = X[:, x_batch_indices] # X.shape[0] x batch_size
    phi_xs = tensor.phi(x_batch) # dim_phi x batch_size
    phi_x_primes = tensor.K_(np.array([all_actions])) @ phi_xs # all_actions.shape[0] x dim_phi x batch_size

    pis_response = pis(x_batch) # all_actions.shape[0] x x_batch_size
    log_pis = torch.log(pis_response) # all_actions.shape[0] x batch_size

    # Compute V(x)'s
    V_x_primes_arr = torch.zeros([all_actions.shape[0], batch_size])
    for u in range(phi_x_primes.shape[0]):
        V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T
    
    # Get costs
    costs = torch.from_numpy(cost(x_batch, np.array([all_actions]))).float() # all_actions.shape[0] x batch_size

    # Compute expectations
    expectation_us = (costs + lamb*log_pis + gamma*V_x_primes_arr) * pis_response # all_actions.shape[0] x batch_size
    expectation_u = torch.sum(expectation_us, axis=0).reshape(-1,1) # (batch_size, 1)

    # Use model to get V(x) for all phi(x)s
    V_xs = model(torch.from_numpy(phi_xs.T).float()) # (batch_size, 1)

    # Compute squared differences
    squared_differences = torch.pow(V_xs - expectation_u, 2) # 1 x batch_size
    total = torch.sum(squared_differences) / batch_size # scalar

    return total

#%%
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

model = torch.nn.Sequential(torch.nn.Linear(phi_dim, 1))
model.apply(init_weights)

# model = torch.load(PATH)

learning_rate = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% Run Algorithm
epochs = 500
epsilon = 1e-2
batch_size = 2**9
batch_scale = 3
bellman_errors = [discrete_bellman_error(batch_size*batch_scale).data.numpy()]
BE = bellman_errors[-1]
print("Initial Bellman error:", BE)

while gamma <= 0.99:
    for epoch in range(epochs):
        # Get random batch of X and Phi_X
        x_batch_indices = np.random.choice(X.shape[1], batch_size, replace=False)
        x_batch = X[:,x_batch_indices] # X.shape[0] x batch_size
        phi_x_batch = tensor.phi(x_batch) # dim_phi x batch_size

        # Compute estimate of V(x) given the current model
        V_x = model(torch.from_numpy(phi_x_batch.T).float()).T # (1, batch_size)

        # Get current distribution of actions for each state
        pis_response = pis(x_batch) # (all_actions.shape[0], batch_size)
        log_pis = torch.log(pis_response) # (all_actions.shape[0], batch_size)

        # Compute V(x)'
        phi_x_primes = tensor.K_(np.array([all_actions])) @ phi_x_batch # all_actions.shape[0] x dim_phi x batch_size
        V_x_primes_arr = torch.zeros([all_actions.shape[0], batch_size])
        for u in range(phi_x_primes.shape[0]):
            V_x_primes_arr[u] = model(torch.from_numpy(phi_x_primes[u].T).float()).T

        # Get costs
        costs = torch.from_numpy(cost(x_batch, np.array([all_actions]))).float() # (all_actions.shape[0], batch_size)

        # Compute expectations
        expectation_term_1 = torch.sum(
            torch.mul(
                (costs + lamb*log_pis + gamma*V_x_primes_arr),
                pis_response
            ),
            dim=0
        ).reshape(1,-1) # (1, batch_size)

        # Equation 2.21 in Overleaf
        loss = torch.sum( torch.pow( V_x - expectation_term_1, 2 ) ) # ()
        
        # Back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Recompute Bellman error
        BE = discrete_bellman_error(batch_size*batch_scale).data.numpy()
        bellman_errors = np.append(bellman_errors, BE)

        # Every so often, print out and save the bellman error(s)
        if (epoch+1) % 500 == 0:
            # np.save('manifold_bellman_errors.npy', bellman_errors)
            torch.save(model, PATH)
            print(f"Bellman error at epoch {epoch+1}: {BE}")

        if BE <= epsilon:
            torch.save(model, PATH)
            break

    gamma += 0.02

#%% Extract
def learned_policy(x):
    with torch.no_grad():
        pis_response = pis(x)[:,0]
    return np.random.choice(all_actions, size=action_column_shape, p=pis_response.data.numpy())

#%% Follow trajectory with initial state from KRONIC paper
kronic_xs = np.zeros([num_steps_per_episode,state_dim])
koopman_xs = np.zeros([num_steps_per_episode,state_dim])

kronic_us = np.zeros([num_steps_per_episode,action_dim])
koopman_us = np.zeros([num_steps_per_episode,action_dim])

kronic_costs = np.zeros([num_steps_per_episode])
koopman_costs = np.zeros([num_steps_per_episode])

x = np.array([
    [-5],
    [5]
])
kronic_x = x
koopman_x = x
for step in range(num_steps_per_episode):
    kronic_xs[step] = kronic_x[:,0]
    koopman_xs[step] = koopman_x[:,0]

    phi_kronic_x = np.append(kronic_x, [[kronic_x[0,0]**2]], axis=0)
    phi_koopman_x = np.append(koopman_x, [[koopman_x[0,0]**2]], axis=0)

    kronic_u = lqr_policy(phi_kronic_x, entropy=False)
    koopman_u = zero_policy(koopman_x) # learned_policy(koopman_x)

    kronic_us[step] = kronic_u
    koopman_us[step] = koopman_u

    kronic_costs[step] = lifted_cost(phi_kronic_x, kronic_u)
    koopman_costs[step] = lifted_cost(phi_koopman_x, koopman_u)

    kronic_x = f(kronic_x, kronic_u)
    koopman_x = f(koopman_x, koopman_u)

data_points_range = np.arange(num_steps_per_episode)

#%% Plot state dynamics
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

#%% Plot action distributions
fig, axs = plt.subplots(2)
fig.suptitle("Action Distributions (KRONIC vs. Koopman)")

axs[0].set_title("KRONIC actions")
axs[0].set(xlabel="Actions", ylabel="Count")

axs[1].set_title("Koopman actions")
axs[1].set(xlabel="Actions", ylabel="Count")

for i in range(action_dim):
    axs[0].hist(kronic_us[:,i])
    axs[1].hist(koopman_us[:,i])

plt.tight_layout()
plt.show()

#%% Plot x_1 vs. x_2
fig, axs = plt.subplots(2)
fig.suptitle("x_1 vs. x_2 (KRONIC vs. Koopman)")

axs[0].set_title("KRONIC")
axs[0].set(xlabel="x_1", ylabel="x_2")

axs[1].set_title("Koopman")
axs[1].set(xlabel="x_1", ylabel="x_2")

axs[0].plot(kronic_xs[:,0], kronic_xs[:,1])
axs[1].plot(koopman_xs[:,0], koopman_xs[:,1])

plt.tight_layout()
plt.show()

#%% Plot costs over time
fig, axs = plt.subplots(2)
fig.suptitle("Costs Over Time (KRONIC vs. Koopman)")

axs[0].set_title("KRONIC")
axs[0].set(xlabel="Step", ylabel="Cost")

axs[1].set_title("Koopman")
axs[1].set(xlabel="Step", ylabel="Cost")

axs[0].plot(data_points_range, kronic_costs)
axs[1].plot(data_points_range, koopman_costs)

plt.tight_layout()
plt.show()

#%% Print average cost over the episode
print(f"Average cost throughout the episode (KRONIC): {kronic_costs.mean()}")
print(f"Average cost throughout the episode (Koopman): {koopman_costs.mean()}")