import matplotlib.pyplot as plt
import numpy as np

# Create plot figure
fig = plt.figure()
# fig = plt.figure(figsize=(16,9))

# Plot bellman errors over time
ax = fig.add_subplot(1, 1, 1)
ax.set_title(f"Bellman Errors During Policy Training")
ax.set_xlabel("Epoch #")
ax.set_ylabel("Bellman Error")
ax.set_ylim(0, 1e6)
# ax.set_ylim(0, 30_000)
bellman_errors = np.load("analysis/tmp/discrete_value_iteration/training_data/bellman_errors.npy")
ax.plot(bellman_errors)

plt.tight_layout()
plt.show()