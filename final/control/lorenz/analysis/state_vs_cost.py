#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#%% Load numpy arrays
folder = "./analysis/tmp"
array_types = ['states', 'actions', 'costs']
controller_types = ['actor_critic', 'value_iteration', 'lqr']
extension = 'npy'
prefix=""
suffix=""

actor_critic_states = np.load(f"{folder}/{prefix}{controller_types[0]}_{array_types[0]}{suffix}.{extension}")
actor_critic_actions = np.load(f"{folder}/{prefix}{controller_types[0]}_{array_types[1]}{suffix}.{extension}")
actor_critic_costs = np.load(f"{folder}/{prefix}{controller_types[0]}_{array_types[2]}{suffix}.{extension}")
value_iteration_states = np.load(f"{folder}/{prefix}{controller_types[1]}_{array_types[0]}{suffix}.{extension}")
value_iteration_actions = np.load(f"{folder}/{prefix}{controller_types[1]}_{array_types[1]}{suffix}.{extension}")
value_iteration_costs = np.load(f"{folder}/{prefix}{controller_types[1]}_{array_types[2]}{suffix}.{extension}")
lqr_states = np.load(f"{folder}/{prefix}{controller_types[2]}_{array_types[0]}{suffix}.{extension}")
lqr_actions = np.load(f"{folder}/{prefix}{controller_types[2]}_{array_types[1]}{suffix}.{extension}")
lqr_costs = np.load(f"{folder}/{prefix}{controller_types[2]}_{array_types[2]}{suffix}.{extension}")

#%% Create figure
fig = plt.figure()

#%% Choose episode number to plot
# Picked final path for demonstration
path_num_to_plot = -1
episode_num_to_plot = 0

#%% Plot individual state dimensions over time
ax = fig.add_subplot(311)
ax.set_title("States over time")
ax.plot(actor_critic_states[path_num_to_plot, episode_num_to_plot, :, 0])
ax.plot(actor_critic_states[path_num_to_plot, episode_num_to_plot, :, 1])
ax.legend(['x_0', 'x_1'])

#%% Plot actions over time
ax = fig.add_subplot(312)
ax.plot(actor_critic_actions[path_num_to_plot, episode_num_to_plot, :, 0])

#%% Plot costs over time
ax = fig.add_subplot(313)
ax.plot(actor_critic_costs[path_num_to_plot, episode_num_to_plot])

#%% Show plots
plt.show()

#%% Compute average cost for each initial state
actor_critic_average_cost_per_initial_state = actor_critic_costs.mean(axis=2)
value_iteration_average_cost_per_initial_state = value_iteration_costs.mean(axis=2)
lqr_average_cost_per_initial_state = lqr_costs.mean(axis=2)

#%% Print minimum average costs
print("Minimum actor critic average cost:", np.min(actor_critic_average_cost_per_initial_state))
print("Minimum value iteration average cost:", np.min(value_iteration_average_cost_per_initial_state))
print("Minimum LQR average cost:", np.min(lqr_average_cost_per_initial_state))

#%% Plot costs vs initial state
fig = plt.figure()

#%% Plot actor-critic policy costs vs initial states
ax = fig.add_subplot(131, projection='3d')
ax.scatter(
    actor_critic_states[:, episode_num_to_plot, 0, 0],
    actor_critic_states[:, episode_num_to_plot, 0, 1],
    actor_critic_states[:, episode_num_to_plot, 0, 2],
    c=actor_critic_average_cost_per_initial_state,
    cmap=plt.hot()
)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("x_2")
plt.savefig(f"{folder}/{controller_types[0]}_costs_vs_initial_state.svg")
plt.savefig(f"{folder}/{controller_types[0]}_costs_vs_initial_state.png")

#%% Plot value iteration policy costs vs initial states
ax = fig.add_subplot(132, projection='3d')
ax.scatter(
    value_iteration_states[:, episode_num_to_plot, 0, 0],
    value_iteration_states[:, episode_num_to_plot, 0, 1],
    value_iteration_states[:, episode_num_to_plot, 0, 2],
    c=value_iteration_average_cost_per_initial_state,
    cmap=plt.hot()
)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("x_2")
# ax.set_zlabel("cost")
plt.savefig(f"{folder}/{controller_types[1]}_costs_vs_initial_state.svg")
plt.savefig(f"{folder}/{controller_types[1]}_costs_vs_initial_state.png")

#%% Plot LQR costs vs initial states
ax = fig.add_subplot(133, projection='3d')
ax.scatter(
    lqr_states[:, episode_num_to_plot, 0, 0],
    lqr_states[:, episode_num_to_plot, 0, 1],
    lqr_states[:, episode_num_to_plot, 0, 2],
    c=lqr_average_cost_per_initial_state,
    cmap=plt.hot()
)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("x_2")
# ax.set_zlabel("cost")
plt.savefig(f"{folder}/{controller_types[2]}_costs_vs_initial_state.svg")
plt.savefig(f"{folder}/{controller_types[2]}_costs_vs_initial_state.png")

#%% Show plots
plt.show()

#%% Plot all pairs of states
fig = plt.figure()

state_dim = lqr_states.shape[-1]
for pair in ((0,1), (0,2), (1,2)):
    i, j = pair
    column_index = i+j

    ax = fig.add_subplot(3, 3, column_index, projection='3d')
    ax.scatter(
        actor_critic_states[:, episode_num_to_plot, 0, i],
        actor_critic_states[:, episode_num_to_plot, 0, j],
        actor_critic_average_cost_per_initial_state
    )
    ax.set_xlabel(f"x_{i}")
    ax.set_ylabel(f"x_{j}")
    ax.set_zlabel("cost")
    
    ax = fig.add_subplot(3, 3, state_dim+column_index, projection='3d')
    ax.scatter(
        value_iteration_states[:, episode_num_to_plot, 0, i],
        value_iteration_states[:, episode_num_to_plot, 0, j],
        value_iteration_average_cost_per_initial_state
    )
    ax.set_xlabel(f"x_{i}")
    ax.set_ylabel(f"x_{j}")
    ax.set_zlabel("cost")

    ax = fig.add_subplot(3, 3, state_dim*2+column_index, projection='3d')
    ax.scatter(
        lqr_states[:, episode_num_to_plot, 0, i],
        lqr_states[:, episode_num_to_plot, 0, j],
        lqr_average_cost_per_initial_state
    )
    ax.set_xlabel(f"x_{i}")
    ax.set_ylabel(f"x_{j}")
    ax.set_zlabel("cost")

#%% Show plot
plt.show()

#%% Return index and value of closest value in array
import math

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1, array[idx-1]
    else:
        return idx, array[idx]

#%% Compare actor critic to lqr cost
ratio = (actor_critic_average_cost_per_initial_state / lqr_average_cost_per_initial_state)[:,0]
ratio_min_index = np.argmin(ratio)
ratio_max_index = np.argmax(ratio)
# mode = stats.mode(ratio).mode[0,0]
median = np.median(ratio)
median_index, _ = find_nearest(ratio, median)
print(median_index)

#%% Plot trajectories
fig = plt.figure()

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot3D(
    actor_critic_states[ratio_min_index, episode_num_to_plot, :, 0],
    actor_critic_states[ratio_min_index, episode_num_to_plot, :, 1],
    actor_critic_states[ratio_min_index, episode_num_to_plot, :, 2]
)
ax.plot3D(
    lqr_states[ratio_min_index, episode_num_to_plot, :, 0],
    lqr_states[ratio_min_index, episode_num_to_plot, :, 1],
    lqr_states[ratio_min_index, episode_num_to_plot, :, 2]
)
ax.set_title("MINIMUM")

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot3D(
    actor_critic_states[ratio_max_index, episode_num_to_plot, :, 0],
    actor_critic_states[ratio_max_index, episode_num_to_plot, :, 1],
    actor_critic_states[ratio_max_index, episode_num_to_plot, :, 2]
)
ax.plot3D(
    lqr_states[ratio_max_index, episode_num_to_plot, :, 0],
    lqr_states[ratio_max_index, episode_num_to_plot, :, 1],
    lqr_states[ratio_max_index, episode_num_to_plot, :, 2]
)
ax.set_title("MAXIMUM")

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot3D(
    actor_critic_states[median_index, episode_num_to_plot, :, 0],
    actor_critic_states[median_index, episode_num_to_plot, :, 1],
    actor_critic_states[median_index, episode_num_to_plot, :, 2],
)
ax.plot3D(
    lqr_states[median_index, episode_num_to_plot, :, 0],
    lqr_states[median_index, episode_num_to_plot, :, 1],
    lqr_states[median_index, episode_num_to_plot, :, 2]
)
ax.set_title("MEDIAN")

plt.show()