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
                event - Tuple to be stored.
        """

        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.pointer] = event
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample(self, batch_size):
        """
            Samples a specified number of events.

            INPUTS:
                batch_size - Int number that determines the amount of events to be sampled.

            OUTPUTS:
                Random list with stored events.
        """

        return random.sample(self.data, batch_size) 

    def retrieve(self):
        """
            Returns the whole stored data

            OUTPUTS:
                data
        """

        return self.data
    
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

    def __init__(self, input_dim):
        """
            Creates the three linear layers of the network.

            INPUTS:  
                input_dim - Int that specifies the size of input.
        """

        super().__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # self.l3.weight.data.uniform_(-3e-3, 3e-3)
        # self.l3.bias.data.uniform_(-3e-3, 3e-3)

        # Define loss function and optimizer
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)    
    
    def forward(self, s):
        """
            Calculates output for the given input.

            INPUTS:
                x - Input to be propagated through the network.

            OUTPUTS:
                x - Number that represents the approximate value of the input.
        """

        x = F.relu(self.l1(s))
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

    def __init__(self, s_dim, a_dim):
        """
            Creates the three linear layers of the network.

            INPUTS:
            input_dim - Int that specifies the size of input.
        """

        super().__init__()
        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # self.l3.weight.data.uniform_(-3e-3, 3e-3)
        # self.l3.bias.data.uniform_(-3e-3, 3e-3) 

        # Define loss function and optimizer
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(self, s,a):
        """
            Calculates output for the given input.

            INPUTS:
                x - Input to be propagated through the network.

            OUTPUTS:
                x - Number that represents the approximate value of the input.
        """

        # Concatenate action vector to state vector
        x = torch.cat([s, a], 1)

        # Pass extended vector through network
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return(x)

class PolicyNetwork(nn.Module):
    '''
    Description:
    The policyNet is a standard fully connected NN with ReLU and sigmoid activation 
    functions and 3 linear layers. This net determines the action for a given state. 

    Attributes:  
    l1,l2,l3 -- linear layers
    
    Methods:
    forward -- calculates otput of network
    '''
    def __init__(self, input_dim, output_dim, min_log_stdev=-30, max_log_stdev=30):
        '''
        Descrption:
        Creates the three linear layers of the net

        Arguments:  
        input_dim -- int that specifies the size of input        
        '''
        super().__init__()
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l31 = nn.Linear(256, output_dim)
        self.l32 = nn.Linear(256, output_dim)

        # self.l31.weight.data.uniform_(-3e-3, 3e-3)
        # self.l32.weight.data.uniform_(-3e-3, 3e-3)
        # self.l31.bias.data.uniform_(-3e-3, 3e-3)
        # self.l32.bias.data.uniform_(-3e-3, 3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = 3e-4)    
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.l31(x)
        log_stdev = self.l32(x)
        log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev
    
    def sample_action(self, s):
        """"
            Calculates output for the given input.

            INPUTS:
                x - input to be propagated through the net.

            OUTPUTS:
                action
        """

        # Get muy and sigma from network
        m, log_stdev = self(s)

        # Sample action from normal distribution
        u = m + log_stdev.exp()*torch.randn_like(m)

        # Normalize action between -1 and 1
        a = torch.tanh(u).cpu()

        return a

    def sample_action_and_llhood(self, s):
        # Get mu and sigma from network
        m, log_stdev = self(s)
        stdev = log_stdev.exp()

        # Sample action from normal distribution
        u = m + stdev*torch.randn_like(m)

        # Normalize action between -1 and 1
        a = torch.tanh(u)

        # From section C of appendix in SAC paper
        llhood = (
            Normal(m, stdev).log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), 1e-6, 1.0))
        ).sum(dim=1, keepdim=True)

        return a, llhood