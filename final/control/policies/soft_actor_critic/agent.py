import numpy as np
import time
import torch

from final.control.policies.soft_actor_critic.networks import Memory, VNetwork, QNetwork, PolicyNetwork
from sys import stdout
from typing import List

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

""" GENERAL METHODS """

def update_net(target, source, tau):
    assert tau >= 0 and tau <= 1, "tau must be between 0 and 1 (inclusive)."

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

# def normalize_angle(x):
#     return (((x+np.pi) % (2*np.pi)) - np.pi)

def scale_action(a, min, max):
    """
        Scale the result of tanh(u) to that of the action space.
    """

    return (0.5*(a+1.0)*(max-min) + min)

""" CLASSES """

class Agent:
    """
        ATTRIBUTES:    

        METHODS:
            fit --  
            s_score --  
            sample_a -- 
            sample_m_state -- 
            act --    
            learn --
    """

    def __init__(
        self,
        state_dim=2,
        action_dim=1,
        memory_capacity=50_000,
        batch_size=64,
        discount_factor=0.99,
        temperature=1.0, # 0.33
        soft_lr=5e-3,
        reward_scale=1.0
    ):
        """
            Initializes the agent.

            INPUTS:

            OUTPUTS:
                None
        """

        self.x_dim = state_dim
        self.u_dim = action_dim
        self.xu_dim = self.x_dim + self.u_dim
        self.batch_size = batch_size 
        self.gamma = discount_factor
        self.soft_lr = soft_lr        
        self.alpha = temperature
        self.reward_scale = reward_scale
         
        self.memory = Memory(memory_capacity)
        self.actor = PolicyNetwork(self.x_dim, self.u_dim).to(device)        
        self.critic1 = QNetwork(self.x_dim, self.u_dim).to(device)
        self.critic2 = QNetwork(self.x_dim, self.u_dim).to(device)
        self.baseline = VNetwork(self.x_dim).to(device)
        self.baseline_target = VNetwork(self.x_dim).to(device)

        update_net(
            target=self.baseline_target,
            source=self.baseline,
            tau=1.0 # 0.001
        )

    def act(self, state):
        with torch.no_grad():
            return self.actor.sample_action(state)

    def memorize(self, event):
        self.memory.store(event[np.newaxis, :])

    def learn(self):
        # Sample batch of data from replay buffer
        data_batch = self.memory.sample(self.batch_size)
        data_batch = np.concatenate(data_batch, axis=0)

        # Extract concatenated batch array into separate arrays
        x_batch = torch.FloatTensor(data_batch[:, :self.x_dim]).to(device)
        u_batch = torch.FloatTensor(data_batch[:, self.x_dim:self.xu_dim]).to(device)
        r_batch = torch.FloatTensor(data_batch[:, self.xu_dim]).unsqueeze(1).to(device)
        x_prime_batch = torch.FloatTensor(data_batch[:, self.xu_dim+1:self.xu_dim+1+self.x_dim]).to(device)

        # Optimize Q networks
        q1 = self.critic1(x_batch, u_batch)
        q2 = self.critic2(x_batch, u_batch)     
        v_prime = self.baseline_target(x_prime_batch)
        q_approx = self.reward_scale * r_batch + self.gamma * v_prime

        q1_loss = self.critic1.loss_func(q1, q_approx.detach())
        self.critic1.optimizer.zero_grad()
        q1_loss.backward()
        self.critic1.optimizer.step()

        q2_loss = self.critic2.loss_func(q2, q_approx.detach())
        self.critic2.optimizer.zero_grad()
        q2_loss.backward()
        self.critic2.optimizer.step()

        # Optimize V network
        v = self.baseline(x_batch)
        a_batch_off, llhood = self.actor.sample_action_and_llhood(x_batch)
        q1_off = self.critic1(x_batch, a_batch_off)
        q2_off = self.critic2(x_batch, a_batch_off)
        q_off = torch.min(q1_off, q2_off)
        v_approx = q_off - self.alpha*llhood

        v_loss = self.baseline.loss_func(v, v_approx.detach())
        self.baseline.optimizer.zero_grad()
        v_loss.backward()
        self.baseline.optimizer.step()
        
        # Optimize policy network
        pi_loss = (llhood - q_off).mean()
        self.actor.optimizer.zero_grad()
        pi_loss.backward()
        self.actor.optimizer.step()

        # Update V target network
        update_net(
            target=self.baseline_target,
            source=self.baseline,
            tau=self.soft_lr
        )

class System:
    def __init__(
        self,
        state_dim: float,
        all_actions: List[List[float]],
        memory_capacity=200_000,
        env_steps=1,
        grad_steps=1,
        init_steps=256,
        reward_scale=25,
        temperature=1.0,
        soft_lr=5e-3,
        batch_size=256,
        hard_start=False,
        original_state=True
    ): # system='Hopper-v2' or 'Pendulum-v0', 'Hopper-v2', 'HalfCheetah-v2', 'Swimmer-v2'
        self.x_dim = state_dim
        self.all_actions = np.sort(all_actions, axis=1)
        self.u_dim = all_actions.shape[1]
        self.xu_dim = self.x_dim + self.u_dim
        self.event_dim = self.x_dim*2 + self.u_dim + 1 # state, action, reward, and next_state in an array

        self.env_steps = env_steps
        self.grad_steps = grad_steps
        self.init_steps = init_steps
        self.batch_size = batch_size
        self.hard_start = hard_start
        self.original_state = original_state

        self.min_action = self.all_actions[:, 0]
        self.max_action = self.all_actions[:, -1]
        self.temperature = temperature
        self.reward_scale = reward_scale

        self.agent = Agent(
            s_dim=self.s_dim,
            a_dim=self.a_dim,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
            reward_scale=reward_scale,
            temperature=temperature,
            soft_lr=soft_lr
        )    

    def initialization(self):
        event = np.empty(self.e_dim)
        if self.hard_start:
            initial_state = np.array([-np.pi,0.0])
            self.env.state = initial_state
        else:
            self.env.reset()
        if self.original_state:
            state = self.env._get_obs()
        else:
            state = self.env.state 
        
        for init_step in range(0, self.init_steps):            
            action = np.random.rand(self.a_dim)*2 - 1
            reward = self.env.step(scale_action(action, self.min_action, self.max_action))[1]
            if self.original_state:
                next_state = self.env._get_obs()
            else:
                next_state = self.env.state
                next_state[0] = normalize_angle(next_state[0])

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.e_dim] = next_state

            self.agent.memorize(event)
            state = np.copy(next_state)
    
    def interaction(self, learn=True, remember=True):   
        event = np.empty(self.e_dim)
        if self.original_state:
            state = self.env._get_obs()
        else:            
            state = self.env.state 
            state[0] = normalize_angle(state[0])

        for env_step in range(0, self.env_steps):
            cuda_state = torch.FloatTensor(state).unsqueeze(0).to(device)         
            action = self.agent.act(cuda_state, explore=learn)

            reward = self.env.step(scale_action(action, self.min_action, self.max_action))[1]

            if self.original_state:
                next_state = self.env._get_obs()
            else:
                next_state = self.env.state
                next_state[0] = normalize_angle(next_state[0])

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.e_dim] = next_state

            if remember:
                self.agent.memorize(event)   

            state = np.copy(next_state)

        if learn:
            for grad_step in range(0, self.grad_steps):
                self.agent.learn()

        return(event)

    def train_agent(self, tr_epsds, epsd_steps, initialization=True):
        if initialization:
            self.initialization()

        min_reward = 1e10
        max_reward = -1e10
        mean_reward = 0.0   
        min_mean_reward = 1e10
        max_mean_reward = -1e10   

        mean_rewards = []           

        for epsd in range(0, tr_epsds):
            epsd_min_reward = 1e10
            epsd_max_reward = -1e10                
            epsd_mean_reward = 0.0

            if self.hard_start:
                initial_state = np.array([-np.pi,0.0])
                self.env.state = initial_state
            else:
                self.env.reset()

            for epsd_step in range(0, epsd_steps):
                if len(self.agent.memory.data) < self.batch_size:
                    event = self.interaction(learn=False)
                else:
                    event = self.interaction()
                r = event[self.sa_dim]

                min_reward = np.min([r, min_reward])
                max_reward = np.max([r, max_reward])
                epsd_min_reward = np.min([r, epsd_min_reward])                        
                epsd_max_reward = np.max([r, epsd_max_reward])                        
                epsd_mean_reward += r           

            # if epsd_mean_reward > max_mean_reward:
            #     pickle.dump(self,open(self.type+'.p','wb'))

            epsd_mean_reward /= epsd_steps            
            mean_rewards.append(epsd_mean_reward)

            min_mean_reward = np.min([epsd_mean_reward, min_mean_reward])
            max_mean_reward = np.max([epsd_mean_reward, max_mean_reward])            
            mean_reward += (epsd_mean_reward - mean_reward)/(epsd+1)            
            stdout.write("Finished epsd %i, epsd.min(r) = %.4f, epsd.max(r) = %.4f, min.(r) = %.4f, max.(r) = %.4f, min.(av.r) = %.4f, max.(av.r) = %.4f, epsd.av.r = %.4f, total av.r = %.4f\r " %
                ((epsd+1), epsd_min_reward, epsd_max_reward, min_reward, max_reward, min_mean_reward, max_mean_reward, epsd_mean_reward, mean_reward))
            stdout.flush()
            time.sleep(0.0001)
        print()

        return mean_rewards