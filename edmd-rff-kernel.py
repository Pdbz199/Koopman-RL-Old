#%% Imports
import algorithms
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

#%% Load data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)

#%% X_tilde
X_tilde = np.append(X, U, axis=0)[:, :1000]
d = X_tilde.shape[0]
m = X_tilde.shape[1]
Y_tilde = np.append(np.roll(X_tilde,-1)[:, :-1], np.zeros((d,1)), axis=1)

#%% RBF Sampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X_tilde)
k = X_features.shape[1]
psi = lambda x: X_features.T @ x.reshape(-1,1)

#%% Nystroem
# from sklearn import datasets, svm
# from sklearn.kernel_approximation import Nystroem
# X, y = datasets.load_digits(n_class=9, return_X_y=True)
# data = X / 16.
# feature_map_nystroem = Nystroem(gamma=.2,
#                                 random_state=1,
#                                 n_components=300)
# data_transformed = feature_map_nystroem.fit_transform(data)

#%% Psi matrices
def getPsiMatrix(psi, X):
    matrix = np.empty((k,m))
    for col in range(m):
        matrix[:, col] = psi(X[:, col])[:, 0]
    return matrix

Psi_X_tilde = getPsiMatrix(psi, X_tilde)
Psi_Y_tilde = getPsiMatrix(psi, Y_tilde)

#%% Koopman
# || Y             - X B               ||
# || Psi_Y_tilde   - K Psi_X_tilde     ||
# || Psi_Y_tilde.T - Psi_X_tilde.T K.T ||
K = algorithms.rrr(Psi_X_tilde.T, Psi_Y_tilde.T, Psi_Y_tilde.shape[0]).T

#%% Find mapping from Psi_X to X
B = algorithms.SINDy(Psi_X_tilde.T, X_tilde.T, X_tilde.shape[0])

#%% Validate
data_point_index = 10000
x_tilde = np.append(X[:,10000].reshape(-1,1), U[:,10000].reshape(-1,1), axis=0)
y_tilde = np.append(X[:,10001].reshape(-1,1), U[:,10001].reshape(-1,1), axis=0)

print(B.T @ K @ psi(x_tilde))
print()
print(y_tilde)

#%% Multiple predictions
horizon = 2
x_tilde = np.append(X[:,data_point_index].reshape(-1,1), U[:,data_point_index].reshape(-1,1), axis=0)
y_tilde = np.append(X[:,data_point_index+horizon].reshape(-1,1), U[:,data_point_index+horizon].reshape(-1,1), axis=0)

predicted = K @ psi(x_tilde)
for i in range(horizon-1):
    predicted = K @ predicted

print(B.T @ predicted)
print()
print(y_tilde)

#%%