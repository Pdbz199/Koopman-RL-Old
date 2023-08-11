import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        try:
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
        except:
            pass
        try:
            torch.nn.init.constant_(m.bias, 0)
        except:
            pass

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.mean_linear1 = nn.Linear(num_inputs, hidden_dim)
        self.mean_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear3 = nn.Linear(hidden_dim, num_actions)

        # self.log_std_linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.log_std_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.log_std_linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2)

    def forward(self, state):
        x = self.mean_linear1(state)
        x = F.relu(x)
        x = self.mean_linear2(x)
        x = F.relu(x)
        mean = self.mean_linear3(x)

        # x = self.log_std_linear1(state)
        # x = F.relu(x)
        # x = self.log_std_linear2(x)
        # x = F.relu(x)
        log_std = self.log_std_linear3(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)