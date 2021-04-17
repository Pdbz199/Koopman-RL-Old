#%%
import numpy as np
import observables
from general_model import GeneratorModel
from cartpole_reward import cartpoleReward

X = (np.load('random-cartpole-states.npy'))[:5000].T # states
U = (np.load('random-cartpole-actions.npy'))[:5000].T # actions
psi = observables.monomials(2)

model = GeneratorModel(psi, cartpoleReward)
model.fit(X, U)

#%%
print(model.sample_action())

# %%
