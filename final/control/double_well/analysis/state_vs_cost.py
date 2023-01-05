#%% Imports
import matplotlib.pyplot as plt
import numpy as np

#%% Load numpy arrays
folder = "./analysis/tmp/test"
array_types = ['states', 'actions', 'costs']
controller_types = ['actor_critic', 'value_iteration', 'lqr']
extension = 'npy'
prefix="TEST-"
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
ax.set_title("Actor Critic x_0 vs x_1")
ax.plot(actor_critic_states[path_num_to_plot, episode_num_to_plot, :, 0])
ax.plot(actor_critic_states[path_num_to_plot, episode_num_to_plot, :, 1])
ax.legend(['x_0', 'x_1'])

#%% Plot actions over time
ax = fig.add_subplot(312)
ax.set_title("Actor Critic Actions")
ax.plot(actor_critic_actions[path_num_to_plot, episode_num_to_plot, :, 0])

#%% Plot costs over time
ax = fig.add_subplot(313)
ax.set_title("Actor Critic Costs")
ax.plot(actor_critic_costs[path_num_to_plot, episode_num_to_plot])

#%% Show plots
plt.tight_layout()
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
ax.set_title("Actor Critic States vs Average Cost")
ax.scatter(actor_critic_states[:, :, 0, 0], actor_critic_states[:, :, 0, 1], actor_critic_average_cost_per_initial_state)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("cost")
plt.savefig(f"{folder}/{controller_types[0]}_costs_vs_initial_state.svg")
plt.savefig(f"{folder}/{controller_types[0]}_costs_vs_initial_state.png")

#%% Plot value iteration policy costs vs initial states
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Value Iteration States vs Average Cost")
ax.scatter(value_iteration_states[:, :, 0, 0], value_iteration_states[:, :, 0, 1], value_iteration_average_cost_per_initial_state)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("cost")
plt.savefig(f"{folder}/{controller_types[1]}_costs_vs_initial_state.svg")
plt.savefig(f"{folder}/{controller_types[1]}_costs_vs_initial_state.png")

#%% Plot LQR costs vs initial states
ax = fig.add_subplot(133, projection='3d')
ax.set_title("LQR States vs Average Cost")
ax.scatter(lqr_states[:, :, 0, 0], lqr_states[:, :, 0, 1], lqr_average_cost_per_initial_state)
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("cost")
plt.savefig(f"{folder}/{controller_types[2]}_costs_vs_initial_state.svg")
plt.savefig(f"{folder}/{controller_types[2]}_costs_vs_initial_state.png")

#%% Show plots
plt.tight_layout()
plt.show()