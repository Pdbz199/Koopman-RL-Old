#%% Imports
import gym
env = gym.make('env:CartPoleControlEnv-v0')
import numpy as np
import torch

# Model variables
state_dim = 4
action_dim = 1
MSE = torch.nn.MSELoss()

#%% Collect dataset(s)
num_episodes = 100
num_steps_per_episode = 100
N = num_episodes * num_steps_per_episode
X = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])
Y = np.zeros([state_dim,N])
for episode in range(num_episodes):
    state = env.reset()
    for step in range(num_steps_per_episode):
        X[:,(episode*num_steps_per_episode)+step] = state
        action = env.action_space.sample()
        U[:,(episode*num_steps_per_episode)+step] = action
        state, _, __, ___ = env.step(action)
        Y[:,(episode*num_steps_per_episode)+step] = state

X = torch.from_numpy(X).float()
U = torch.from_numpy(U).float()
Y = torch.from_numpy(Y).float()

#%% Phi dictionary models
phi_dim = 16

# Encoder maps x to phi(x)
phi_encoder = torch.nn.Sequential(
    torch.nn.Linear(state_dim, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, phi_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(phi_dim, phi_dim)
)
phi_encoder_optimizer = torch.optim.Adam(phi_encoder.parameters(), lr=0.003)

# Decoder maps phi(x) to x
phi_decoder = torch.nn.Sequential(
    torch.nn.Linear(phi_dim, phi_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(phi_dim, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, state_dim)
)
phi_decoder_optimizer = torch.optim.Adam(phi_decoder.parameters(), lr=0.003)

#%% Psi dictionary models
psi_dim = 16

# Encoder maps u to psi(u)
psi_encoder = torch.nn.Sequential(
    torch.nn.Linear(action_dim, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, psi_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(psi_dim, psi_dim)
)
psi_encoder_optimizer = torch.optim.Adam(psi_encoder.parameters(), lr=0.003)

# Decoder maps psi(u) to u
psi_decoder = torch.nn.Sequential(
    torch.nn.Linear(psi_dim, psi_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(psi_dim, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, action_dim)
)
psi_decoder_optimizer = torch.optim.Adam(psi_decoder.parameters(), lr=0.003)

#%% First run
with torch.no_grad():
    Phi_X = phi_encoder(X.T)
    Psi_U = psi_encoder(U.T)
    Phi_Y = phi_encoder(Y.T)
kronMatrix = torch.zeros([psi_dim * phi_dim, N])
for i in range(N):
    kronMatrix[:,i] = torch.kron(
        Psi_U[i],
        Phi_X[i]
    )
M = torch.linalg.lstsq(kronMatrix.T, Phi_Y)[0].T

state = env.reset()
action = env.action_space.sample()
true_next_state, _, __, ___ = env.step(action)
with torch.no_grad():
    kron = torch.kron(
        psi_encoder(torch.from_numpy(action).float()),
        phi_encoder(torch.from_numpy(state).float())
    ).reshape(-1,1)
predicted_next_state = M @ kron
print("Initial error:", MSE(torch.from_numpy(true_next_state), predicted_next_state))

#%%
"""
What is the objective?
1) phi_decoder(phi_encoder(x)) => x
2) psi_decoder(psi_encoder(u)) => u
3) M @ kron(psi_encoder(u), phi_encoder(x)) => phi_encoder(x')
4) decoder(M @ kron(psi_encoder(u), phi_encoder(x))) => x'
"""

#%% Train
num_epochs = 3000
for epoch in range(num_epochs):
    # 1)
    # update phi encoder/decoder
    X_T = phi_decoder(phi_encoder(X.T))
    phi_encoder_decoder_loss = MSE(X_T, X.T)

    phi_encoder_optimizer.zero_grad()
    phi_decoder_optimizer.zero_grad()
    phi_encoder_decoder_loss.backward()
    phi_encoder_optimizer.step()
    phi_decoder_optimizer.step()

    # 2)
    # update psi encoder/decoder
    U_T = psi_decoder(psi_encoder(U.T))
    psi_encoder_decoder_loss = MSE(U_T, U.T)

    psi_encoder_optimizer.zero_grad()
    psi_decoder_optimizer.zero_grad()
    psi_encoder_decoder_loss.backward()
    psi_encoder_optimizer.step()
    psi_decoder_optimizer.step()
    

    # 3)
    Psi_U = psi_encoder(U.T)
    Phi_X = phi_encoder(X.T)
    kronMatrix = torch.zeros([psi_dim * phi_dim, N])
    for i in range(N):
        kronMatrix[:,i] = torch.kron(
            Psi_U[i],
            Phi_X[i]
        )
    with torch.no_grad():
        Phi_Y = phi_encoder(Y.T)
    M = torch.linalg.lstsq(kronMatrix.T, Phi_Y)[0].T
    predicted_Phi_Y = M @ kronMatrix
    linear_dynamics_loss = MSE(predicted_Phi_Y, Phi_Y.T)
    
    phi_encoder_optimizer.zero_grad()
    psi_encoder_optimizer.zero_grad()
    linear_dynamics_loss.backward()
    phi_encoder_optimizer.step()
    psi_encoder_optimizer.step()

    # 4)
    with torch.no_grad():
        Psi_U = psi_encoder(U.T)
        Phi_X = phi_encoder(X.T)
        kronMatrix = torch.zeros([psi_dim * phi_dim, N])
        for i in range(N):
            kronMatrix[:,i] = torch.kron(
                Psi_U[i],
                Phi_X[i]
            )
        Phi_Y = phi_encoder(Y.T)
    M = torch.linalg.lstsq(kronMatrix.T, Phi_Y)[0].T
    predicted_Phi_Y = M @ kronMatrix
    Y_T = phi_decoder(predicted_Phi_Y.T)
    prediction_loss = MSE(Y_T, Y.T)

    phi_decoder_optimizer.zero_grad()
    prediction_loss.backward()
    phi_decoder_optimizer.step()

    #
    if epoch == 0 or (epoch+1) % 25 == 0:
        print("Epoch", epoch+1)

#%% Test
with torch.no_grad():
    Phi_X = phi_encoder(X.T)
    Psi_U = psi_encoder(U.T)
    Phi_Y = phi_encoder(Y.T)
kronMatrix = torch.zeros([psi_dim * phi_dim, N])
for i in range(N):
    kronMatrix[:,i] = torch.kron(
        Psi_U[i],
        Phi_X[i]
    )
M = torch.linalg.lstsq(kronMatrix.T, Phi_Y)[0].T

state = env.reset()
action = env.action_space.sample()
true_next_state, _, __, ___ = env.step(action)
with torch.no_grad():
    kron = torch.kron(
        psi_encoder(torch.from_numpy(action).float()),
        phi_encoder(torch.from_numpy(state).float())
    ).reshape(-1,1)
predicted_next_state = M @ kron
print("Final error:", MSE(torch.from_numpy(true_next_state), predicted_next_state))

#%% Save models
torch.save(phi_encoder, 'dictionary-models/phi-encoder.pt')
torch.save(phi_decoder, 'dictionary-models/phi-decoder.pt')
torch.save(psi_encoder, 'dictionary-models/psi-encoder.pt')
torch.save(psi_decoder, 'dictionary-models/psi-decoder.pt')

#%%