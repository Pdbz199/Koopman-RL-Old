#%% Imports
import gym
import numpy as np
import estimate_L
import algorithmsv2
env = gym.make("Taxi-v3")

def l2_norm(true_state, predicted_state):
    return np.sum( np.power( ( true_state - predicted_state ), 2 ) )

#%% Load data
X = np.load('random-agent/taxi/states.npy').T
Y = np.append(np.roll(X, -1, axis=1)[:,:-1], np.zeros((X.shape[0],1)), axis=1)
U = np.load('random-agent/taxi/actions.npy').reshape(1,-1)

#%% Enumerate all possible states and actions
num_rows = 5
num_cols = 5
num_locations = 5
num_destinations = 4
enumerated_states = []
for i in range(num_rows):
    for j in range(num_cols):
        for k in range(num_locations):
            for l in range(num_destinations):
                state = (i,j,k,l)
                enumerated_states.append(state)
enumerated_states = np.array(enumerated_states)

num_unique_actions = 6
enumerated_actions = []
for i in range(num_unique_actions):
    enumerated_actions.append([i])
enumerated_actions = np.array(enumerated_actions)

#%% Cost function
def reward(x, u):
    u = u[0]
    
    if u == 4:  # Pickup
        # If passenger not at location
        if x[2] == 4 or (x[0], x[1]) != env.locs[x[2]]:
            return -10
    elif u == 5:  # Dropoff
        # if can drop off passenger
        if ((x[0], x[1]) == env.locs[x[3]]) and x[2] == 4:
            return 20
        else: # Drop off at wrong location
            return -10

    return -1 # Default reward when there is no pickup/dropoff

def cost(x, u):
    return -reward(x, u)

#%% Koopman tensor EDMD setup
def phi(x):
    phi_x = np.zeros(enumerated_states.shape[0])

    for i,state in enumerate(enumerated_states):
        if np.array_equal(x, state):
            phi_x[i] = 1
            return phi_x

def psi(u):
    psi_u = np.zeros(num_unique_actions)
    psi_u[u] = 1
    return psi_u

def getPhiMatrix(X):
    Phi_X = []
    for x in X.T:
        Phi_X.append(phi(x))

    return np.array(Phi_X).T

def getPsiMatrix(U):
    Psi_U = []
    for u in U.T:
        Psi_U.append(psi(u))

    return np.array(Psi_U).T

#%% Build Phi and Psi Matrices
Phi_X = getPhiMatrix(X)
Psi_U = getPsiMatrix(U)
print(Phi_X.shape)
print(Psi_U.shape)

num_lifted_state_observations = Phi_X.shape[1]
num_lifted_state_features = Phi_X.shape[0]
num_lifted_action_observations = Psi_U.shape[1]
num_lifted_action_features = Psi_U.shape[0]

# @nb.njit(fastmath=True)
def getPsiPhiMatrix(Psi_U, Phi_X):
    psiPhiMatrix = np.empty((num_lifted_action_features * num_lifted_state_features, num_lifted_state_observations))

    for i in range(num_lifted_state_observations):
        kron = np.kron(Psi_U[:,i], Phi_X[:,i])
        psiPhiMatrix[:,i] = kron

    return psiPhiMatrix

psiPhiMatrix = getPsiPhiMatrix(Psi_U, Phi_X)
print("PsiPhiMatrix shape:", psiPhiMatrix.shape)

#%% Compute M as in writeup
M = estimate_L.rrr(psiPhiMatrix.T, getPhiMatrix(Y).T).T
print("M shape:", M.shape)
assert M.shape == (num_lifted_state_features, num_lifted_state_features * num_lifted_action_features)

#%% Reshape M into Koopman tensor
K = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M.shape[0]):
    K[i] = M[i].reshape((num_lifted_state_features, num_lifted_action_features))
print("K shape:", K.shape)

def K_u(K, u):
    return np.einsum('ijz,z->ij', K, psi(u))

#%% Test prediction error
episodes = 1
norms = []
for episode in range(episodes):
    true_state = env.reset()
    taxi_y,taxi_x,passenger,destination = env.decode(true_state)
    decoded_true_state = np.array([taxi_y, taxi_x, passenger, destination])

    done = False
    while not done:
        action = env.action_space.sample()
        print("Action:", action)

        print("Decoded state:", decoded_true_state)
        phi_x_prime = K_u(K, action) @ phi(decoded_true_state)
        predicted_state = enumerated_states[np.argmax(phi_x_prime)]
        print("Predicted state:", predicted_state)
        true_state, _, done, __ = env.step(action)

        taxi_y,taxi_x,passenger,destination = env.decode(true_state)
        decoded_true_state = np.array([taxi_y, taxi_x, passenger, destination])
        print("Decoded state:", decoded_true_state)

        norms.append(l2_norm(decoded_true_state, predicted_state))

env.close()

norms = np.array(norms)
print("Mean norm:", np.mean(norms))

# pi = algorithmsv2.algorithm2(X, U, phi, psi, K, cost)
# print(pi(U[:,0], X[:,0]))