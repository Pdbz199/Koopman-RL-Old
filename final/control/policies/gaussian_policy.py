import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(0.0)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = np.finfo(np.float32).eps.item()

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        layer_1_dim=256,
        layer_2_dim=128,
        action_space=None
    ):
        super(GaussianPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # self.linear1 = nn.Linear(state_dim, layer_1_dim)
        # self.linear2 = nn.Linear(layer_1_dim, layer_2_dim)
        # self.mean_linear = nn.Linear(layer_2_dim, action_dim)
        # self.log_std_linear = nn.Linear(layer_2_dim, action_dim)
        self.mean_linear = nn.Linear(state_dim, action_dim)
        self.log_std_linear = nn.Linear(state_dim, action_dim)

        self.apply(init_weights)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        # state = torch.Tensor(state)[:, 0]
        x = torch.Tensor(state)

        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def get_action(self, state):
        state = torch.Tensor(state)[:, 0]

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(sample_shape=(self.action_dim,))  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob#, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)