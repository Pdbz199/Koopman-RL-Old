#%%
import numpy as np

from tensor import KoopmanTensor

import sys
sys.path.append('../')
import observables

#%% Datasets
X = np.array([
    [1, 2]
])
Y = np.array([
    [2, 1]
])
U = np.array([
    [1, 0]
])

#%% Use KoopmanTensor object to create a model of the system
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(2),
    psi=observables.monomials(1)
)

#%%
