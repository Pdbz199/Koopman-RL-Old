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

class VNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(VNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.linear2(x)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.linear3(x)
        return x

class KoopmanVNetwork(nn.Module):
    def __init__(self, koopman_tensor):
        super(KoopmanVNetwork, self).__init__()

        self.koopman_tensor = koopman_tensor
        self.phi_state_dim = self.koopman_tensor.Phi_X.shape[0]

        self.linear = nn.Linear(self.phi_state_dim, 1, bias=False)

        self.apply(weights_init_)

        # Optimal parameters for LinearSystem
        # self.linear.weight.data = torch.tensor([[
        #     -0.0020, # 1
        #     0.1184,  # x
        #     0.4977,  # y
        #     -0.0444, # z
        #     1.3893,  # x^2
        #     0.3182,  # x*y
        #     0.1925,  # x*z
        #     0.4038,  # y^2
        #     0.7556,  # y*z
        #     1.2154   # z^2
        # ]])

    def forward(self, state):
        batch_size = state.shape[0]
        phi_xs = torch.zeros((batch_size, self.phi_state_dim))
        for i in range(batch_size):
            x = state[i].view(state.shape[1], 1)
            phi_xs[i] = self.koopman_tensor.phi(x)[:,0]

        output = self.linear(phi_xs)

        return output

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x = self.linear1(xu)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.linear2(x)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.linear3(x)

        return x

class DoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DoubleQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        # Q1
        x1 = self.linear1(xu)
        x1 = F.relu(x1)
        # x1 = F.tanh(x1)
        x1 = self.linear2(x1)
        x1 = F.relu(x1)
        # x1 = F.tanh(x1)
        x1 = self.linear3(x1)

        # Q2
        x2 = self.linear4(xu)
        x2 = F.relu(x2)
        # x2 = F.tanh(x2)
        x2 = self.linear5(x2)
        x2 = F.relu(x2)
        # x2 = F.tanh(x2)
        x2 = self.linear6(x2)

        return x1, x2

class KoopmanQNetwork(nn.Module):
    def __init__(self, koopman_tensor):
        super(KoopmanQNetwork, self).__init__()

        self.koopman_tensor = koopman_tensor
        self.phi_state_dim = self.koopman_tensor.Phi_X.shape[0]

        self.linear = nn.Linear(self.phi_state_dim, 1, bias=False)

        self.apply(weights_init_)

    def forward(self, state, action):
        # E_V(x) = w^T @ K^(u) @ phi(x) = w^T @ E_phi(x')
        # Q(x, u) = r + gamma*E_V(x)
        #         = r + gamma*(w^T @ K^(u) @ phi(x))
        #         = r + gamma*(w^T @ E_phi(x'))

        # Replace QNetwork output with the following:
        # Q(x, u) = r + gamma*(w.T @ tensor.phi_f(x, u))

        batch_size = state.shape[0]
        expected_phi_x_primes = torch.zeros((batch_size, self.phi_state_dim))
        for i in range(batch_size):
            x = state[i].view(state.shape[1], 1)
            u = action[i].view(action.shape[1], 1)
            expected_phi_x_primes[i] = self.koopman_tensor.phi_f(x, u)[:, 0]

        output = self.linear(expected_phi_x_primes)

        return output

class KoopmanDoubleQNetwork(nn.Module):
    def __init__(self, koopman_tensor):
        super(KoopmanDoubleQNetwork, self).__init__()

        self.koopman_tensor = koopman_tensor
        self.phi_state_dim = self.koopman_tensor.Phi_X.shape[0]

        self.linear1 = nn.Linear(self.phi_state_dim, 1, bias=False)
        self.linear2 = nn.Linear(self.phi_state_dim, 1, bias=False)

        self.apply(weights_init_)

    def forward(self, state, action):
        # E_V(x) = w^T @ K^(u) @ phi(x) = w^T @ E_phi(x')
        # Q(x, u) = r + gamma*E_V(x)
        #         = r + gamma*(w^T @ K^(u) @ phi(x))
        #         = r + gamma*(w^T @ E_phi(x'))

        # Replace QNetwork output with the following:
        # Q(x, u) = r + gamma*(w.T @ tensor.phi_f(x, u))

        batch_size = state.shape[0]
        expected_phi_x_primes = torch.zeros((batch_size, self.phi_state_dim))
        for i in range(batch_size):
            x = state[i].view(state.shape[1], 1)
            u = action[i].view(action.shape[1], 1)
            expected_phi_x_primes[i] = self.koopman_tensor.phi_f(x, u)[:, 0]

        output1 = self.linear1(expected_phi_x_primes)
        output2 = self.linear2(expected_phi_x_primes)

        return output1, output2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.mean_linear1 = nn.Linear(state_dim, hidden_dim)
        self.mean_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear3 = nn.Linear(hidden_dim, action_dim)

        # self.log_std_linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.log_std_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.log_std_linear3 = nn.Linear(hidden_dim, action_dim)

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
        # x = F.tanh(x)
        x = self.mean_linear2(x)
        x = F.relu(x)
        # x = F.tanh(x)
        mean = self.mean_linear3(x)

        # x = self.log_std_linear1(state)
        # x = F.relu(x)
        # x = F.tanh(x)
        # x = self.log_std_linear2(x)
        # x = F.relu(x)
        # x = F.tanh(x)
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

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.linear2(x)
        x = F.relu(x)
        # x = F.tanh(x)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)