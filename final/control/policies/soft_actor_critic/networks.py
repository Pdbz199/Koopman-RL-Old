import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

class Memory:
    """
        The Memory class allows to store and sample events.

        ATTRIBUTES:
            capacity - Max amount of events stored.
            data - List with events memorized.
            pointer - Position of the list in which an event will be registered.

        METHODS:
            store - Save one event in "data" in the position indicated by "pointer".
            sample - Returns a uniformly sampled batch of stored events.
            retrieve - Returns the whole information memorized.
            forget - Eliminates all data stored.
    """

    def __init__(self, capacity=50_000):
        """
            Initializes an empty data list and a pointer located at 0.
            Also determines the capacity of the data list.

            INPUTS:
                Capacity - Positive int number.
        """

        self.capacity = capacity
        self.data = []
        self.pointer = 0

    def store(self, event):
        """
            Stores the input event in the location designated by the pointer.
            The pointer is increased by one modulo the capacity.

            INPUTS:
                event - Array to be stored.
        """

        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.pointer] = event
        self.pointer = (self.pointer + 1) % self.capacity # Have pointer wrap around if needed

    def sample(self, batch_size):
        """
            Samples a specified number of events.

            INPUTS:
                batch_size - Int number that determines the amount of events to be sampled.

            OUTPUTS:
                Random list with stored events.
        """

        return random.sample(self.data, batch_size)

    def forget(self):
        """
            Resets the stored data and the pointer.
        """

        self.data = []
        self.pointer = 0

class VNetwork(nn.Module):
    """
        The VNetwork is a standard fully connected NN with ReLU activation functions
        and 3 linear layers that approximates the V function.

        ATTRIBUTES:
            l1, l2, l3 -- Linear layers.

        METHODS:
            forward - Calculates otput of network.
    """

    def __init__(self, state_dim):
        """
            Creates the three linear layers of the network.

            INPUTS:
                state_dim - Int that specifies the size of input state.
        """

        # Initialize parent module
        super().__init__()

        # Define the layers of the network
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Set initial parameter values
        # self.l3.weight.data.uniform_(-3e-3, 3e-3)
        # self.l3.bias.data.uniform_(-3e-3, 3e-3)

        # Define loss function and optimizer
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, state):
        """
            Calculates output for the given input.

            INPUTS:
                x - Input to be propagated through the network.

            OUTPUTS:
                x - Number that represents the approximate value of the input.
        """

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return(x)

class QNetwork(nn.Module):
    """
        Description:
            The QNetwork is a standard fully-connected NN with ReLU activation functions
            and 3 linear layers that approximates the Q function.

        ATTRIBUTES:
            l1, l2, l3 - Linear layers.

        METHODS:
            forward - Calculates otput of network.
    """

    def __init__(self, state_dim, action_dim):
        """
            Creates the three linear layers of the network.

            INPUTS:
            input_dim - Int that specifies the size of input.
        """

        # Initialize parent module
        super().__init__()

        # Define the layers of the network
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Set initial parameter values
        # self.l3.weight.data.uniform_(-3e-3, 3e-3)
        # self.l3.bias.data.uniform_(-3e-3, 3e-3)

        # Define loss function and optimizer
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, state, action):
        """
            Calculates output for the given input.

            INPUTS:
                x - Input to be propagated through the network.

            OUTPUTS:
                x - Number that represents the approximate value of the input.
        """

        # Concatenate action vector to state vector
        x = torch.cat([state, action], axis=1)

        # Pass extended vector through network
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return(x)

class PolicyNetwork(nn.Module):
    """:
        The PolicyNetwork is a standard fully connected NN with ReLU activation 
        functions and 3 linear layers. This net determines the action for a given state. 

        ATTRIBUTES:
            l1, l2, l3 - Linear layers.

        METHODS:
            forward - Calculates output of network.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        min_log_sigma=-30,
        max_log_sigma=30
    ):
        """
            Creates the three linear layers of the network.

            INPUTS:
                input_dim - Int that specifies the size of input.
        """

        # Initialize parent module
        super().__init__()

        # Set the minimum and maximum log std
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        # Define the layers of the network
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l31 = nn.Linear(256, action_dim)
        self.l32 = nn.Linear(256, action_dim)

        # Set initial parameter values
        # self.l31.weight.data.uniform_(-3e-3, 3e-3)
        # self.l32.weight.data.uniform_(-3e-3, 3e-3)
        # self.l31.bias.data.uniform_(-3e-3, 3e-3)
        # self.l32.bias.data.uniform_(-3e-3, 3e-3)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        # Get mu and log sigma
        mu = self.l31(x)
        log_sigma = self.l32(x)

        # Clamp log sigma to be within the defined range
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)

        return mu, log_sigma

    def sample_action(self, state):
        """"
            Calculates output for the given input.

            INPUTS:
                state - Input to be propagated through the network.

            OUTPUTS:
                action
        """

        # Get mu and sigma from network
        mu, log_sigma = self(state)

        # Sample action from normal distribution
        u = mu + log_sigma.exp()*torch.randn_like(mu)

        # Normalize action between -1 and 1
        a = torch.tanh(u).cpu()

        return a

    def sample_action_and_log_probability(self, state):
        # Get mu and sigma from network
        mu, log_sigma = self(state)
        sigma = log_sigma.exp()

        # Sample action from normal distribution
        u = mu + sigma*torch.randn_like(mu)

        # Normalize action between -1 and 1
        a = torch.tanh(u)

        # From section C of appendix in SAC paper
        log_probability = (
            Normal(mu, sigma).log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), 1e-6, 1.0))
        ).sum(dim=1, keepdim=True)

        return a, log_probability