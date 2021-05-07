import gym
import torch
from pytorch_mppi import mppi

# Create controller with chosen parameters
# ctrl = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
#                          lambda_=lambda_, device=d, 
#                          u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
#                          u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))

# Run in environment
# env = gym.make('CartPole-v0')
# observation = env.reset()
# for i in range(100):
#     action = ctrl.command(observation)
#     observation, reward, done, _ = env.step(action.cpu().numpy())