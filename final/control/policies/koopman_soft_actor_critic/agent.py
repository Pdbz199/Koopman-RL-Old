"""
    LINK TO PAPER:
    https://cloudflare-ipfs.com/ipfs/bafybeibg7jh2hm4w2vcpkgb55qeguo5oiom7qff5s4dxw4xfrjjlamfbg4/Soft%20Actor-Critic-%20Off-Policy%20Maximum%20Entropy%20Deep%20Reinforcement%20Learning%20with%20a%20Stochastic%20Actor.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from final.control.policies.koopman_soft_actor_critic.networks import (
    Memory,
    PolicyNetwork,
    QNetwork,
    VNetwork
)
from final.tensor import KoopmanTensor
from typing import Callable, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" GENERAL METHODS """

def update_net(source, target, tau):
    """
        This is equivalent to line 12 of the Soft Actor-Critic algorithm in the paper.
        (c) in figure 3 of the paper suggests that tau should be between 0.01 and 0.001.
    """

    # Confirm that tau is between 0 and 1
    assert tau >= 0 and tau <= 1, "tau must be between 0 and 1 (inclusive)."

    # Update parameters
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

""" CLASSES """

class Agent:
    """
        ATTRIBUTES:

        METHODS:
            fit - 
            s_score - 
            sample_a - 
            sample_m_state - 
            act - 
            learn - 
    """

    def __init__(
        self,
        koopman_model: KoopmanTensor,
        reward,
        action_minimums,
        action_maximums,
        memory_capacity=50_000,
        batch_size=64,
        discount_factor=0.99,
        reward_scale=10,
        tau=1e-3,
        learning_rate=3e-4
    ):
        """
            Initializes the agent.

            INPUTS:


            OUTPUTS:
                None
        """

        self.action_minimums = action_minimums
        self.action_maximums = action_maximums

        self.koopman_model = koopman_model
        self.reward = reward
        self.x_dim = self.koopman_model.x_dim
        self.u_dim = len(action_minimums)
        self.xu_dim = self.x_dim + self.u_dim
        self.event_dim = self.x_dim + self.u_dim + 1 + self.x_dim + 1 # state, action, reward, next state, and done
        self.batch_size = batch_size
        self.gamma = discount_factor
        self.reward_scale = reward_scale
        self.tau = tau
        self.learning_rate = learning_rate

        self.memory = Memory(memory_capacity)
        self.baseline = VNetwork(
            koopman_model=self.koopman_model,
            learning_rate=self.learning_rate
        ).to(device)
        self.baseline_target = VNetwork(
            koopman_model=self.koopman_model,
            learning_rate=self.learning_rate
        ).to(device)
        self.critic = QNetwork(
            koopman_model=self.koopman_model,
            reward=self.reward,
            learning_rate=self.learning_rate
        ).to(device)
        self.actor = PolicyNetwork(
            self.x_dim,
            action_minimums,
            action_maximums,
            learning_rate=self.learning_rate
        ).to(device)

        # Make sure parameters are initially equal for baseline and baseline_target
        update_net(
            target=self.baseline_target,
            source=self.baseline,
            tau=1.0
        )

    def act(self, state, reparameterize=False):
        """
            Sample scaled action from agent.
            If `reparameterize` is True, compute gradients.
            If `reparameterize` is False, do not compute gradients.
        """

        return self.actor.sample_action(state, reparameterize=reparameterize)

    def memorize(self, event):
        """
            Store an event in the replay buffer.

            INPUTS:
                event - Array containing environment step information.

            OUTPUTS:
                None
        """

        self.memory.store(event[np.newaxis, :])

    def learn(self):
        # Sample batch of data from replay buffer and concatenate data into long arrays
        data_batch = self.memory.sample(self.batch_size)
        data_batch = np.concatenate(data_batch, axis=0)

        # Extract concatenated batch array into separate arrays
        x_batch = torch.FloatTensor(data_batch[:, :self.x_dim]).to(device)
        u_batch = torch.FloatTensor(data_batch[:, self.x_dim:self.xu_dim]).to(device)
        r_batch = torch.FloatTensor(data_batch[:, self.xu_dim]).unsqueeze(1).to(device)
        x_prime_batch = torch.FloatTensor(data_batch[:, self.xu_dim+1:self.event_dim-1]).to(device)
        done = torch.BoolTensor(data_batch[:, self.event_dim-1]).to(device)

        """ Compute Vs, actions, log probabilities, and Qs """

        # Compute V(x) for x in sample from replay buffer
        v_batch = self.baseline(x_batch)

        # Sample actions and log probabilities
        action_batch, log_probability_batch = self.actor.sample_action_and_log_probability(
            x_batch,
            reparameterize=False
        )

        # Compute Q(x, u) for x in sample from replay buffer and action from current policy
        q_new_policy_batch = self.critic(x_batch, action_batch)

        """ Optimize V network """

        # Equation 3
        v_target_batch = q_new_policy_batch - log_probability_batch

        # Compute loss (mean squared error as in equation 5)
        v_loss = 0.5 * self.baseline.loss_func(v_batch, v_target_batch)

        # Backprop
        self.baseline.optimizer.zero_grad()
        v_loss.backward()
        self.baseline.optimizer.step()

        """ Optimize Q networks """

        # Compute V(x') as in equation 8
        v_prime_batch = self.baseline_target(x_prime_batch)
        v_prime_batch[done] = 0.0

        # Compute Q values with actions sampled from replay buffer
        q_old_policy_batch = self.critic(x_batch, u_batch)

        # Equation 8
        q_target = self.reward_scale*r_batch + self.gamma*v_prime_batch

        # Compute loss (Mean squared error as in equation 7)
        q_loss = 0.5 * self.critic.loss_func(q_old_policy_batch, q_target.detach())

        # Backprop
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()

        """ Optimize policy network (equations 10 and 12) """

        # Sample new actions and log probabilities
        action_batch, log_probability_batch = self.actor.sample_action_and_log_probability(
            x_batch,
            reparameterize=True
        )
        q_new_policy_batch = self.critic(x_batch, action_batch)

        # Compute loss
        pi_loss = (log_probability_batch - q_new_policy_batch).mean()

        # Backprop
        self.actor.optimizer.zero_grad()
        pi_loss.backward()
        self.actor.optimizer.step()

        """ Update V target network  """

        update_net(
            target=self.baseline_target,
            source=self.baseline,
            tau=self.tau
        )

class System:
    def __init__(
        self,
        is_gym_env: bool,
        true_dynamics: Callable[[List[float], List[float]], List[float]],
        koopman_model: KoopmanTensor,
        reward: Callable[[List[float], List[float]], float],
        state_minimums: List[float],
        state_maximums: List[float],
        action_minimums: List[float],
        action_maximums: List[float],
        render_env=False,
        memory_capacity=200_000,
        environment_steps=1,
        gradient_steps=1,
        init_steps=256,
        reward_scale=10,
        tau=5e-3,
        learning_rate=3e-4,
        batch_size=256,
        is_episodic=False
    ):
        assert len(state_minimums) == len(state_maximums), "Min and max state must have same dimension."
        assert len(action_minimums) == len(action_maximums), "Min and max action must have same dimension."

        self.is_gym_env = is_gym_env
        self.render_env = render_env
        self.has_completed = False
        self.true_dynamics = true_dynamics
        self.koopman_model = koopman_model
        self.reward = reward

        self.x_dim = len(state_minimums)
        self.u_dim = len(action_minimums)
        self.xu_dim = self.x_dim + self.u_dim
        self.event_dim = self.x_dim + self.u_dim + 1 + self.x_dim + 1 # state, action, reward, next_state, and done

        self.environment_steps = environment_steps
        self.gradient_steps = gradient_steps
        self.init_steps = init_steps
        self.batch_size = batch_size

        self.state_minimums = state_minimums
        self.state_maximums = state_maximums
        self.action_minimums = action_minimums
        self.action_maximums = action_maximums

        self.agent = Agent(
            koopman_model=koopman_model,
            reward=reward,
            action_minimums=action_minimums,
            action_maximums=action_maximums,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
            reward_scale=reward_scale,
            tau=tau,
            learning_rate=learning_rate
        )

        self.is_episodic = is_episodic

        # Set up random number generator
        self.rng = np.random.default_rng()

        self.latest_state = None
    
    def reset_env(self):
        if self.is_gym_env:
            self.latest_state = self.true_dynamics.reset()
        else:
            self.latest_state = (0.5 * (self.rng.random((self.x_dim,1))*2-1) * (self.state_maximums-self.state_minimums))[:, 0]

    def initialization(self):
        # If episodic, reset latest state to random initial condition
        if self.is_episodic or self.latest_state is None:
            self.reset_env()

        state = np.vstack(self.latest_state) # Turn array into column vector

        for init_step in range(self.init_steps):
            # Create empty events matrix
            event = np.empty(self.event_dim)

            # Select random action between -1 and 1 and scale it
            normalized_action = np.random.rand(self.u_dim)*2 - 1
            action = self.agent.actor.scale_action(normalized_action)

            if self.is_gym_env:
                # Get next state, reward, and more from true dynamics
                # next_state, reward, self.has_completed, _ = self.true_dynamics.step(int(action[0])) # Cartpole
                next_state, reward, self.has_completed, _ = self.true_dynamics.step(action) # Bipedal Walker
            else:
                numpy_action = np.vstack(action.numpy())

                # Get reward for state-action pair
                reward = self.reward(state, numpy_action)

                # Get next state from true dynamics
                next_state = self.true_dynamics(state, numpy_action)[:, 0]

            # Populate the event array with state, action, reward, and next state
            event[:self.x_dim] = state[:, 0]
            event[self.x_dim:self.xu_dim] = action
            event[self.xu_dim] = reward
            event[self.xu_dim+1:self.event_dim-1] = next_state
            event[self.event_dim-1] = self.has_completed

            # Add the event to the replay buffer
            self.agent.memorize(event)

            # Update state
            state = np.vstack(next_state)
            self.latest_state = state[:, 0]

            # If done condition is met, reset env for initialization
            if self.has_completed:
                self.reset_env()

    def interaction(self, learn=True, remember=True):
        # Create empty events matrix
        events = np.empty((self.environment_steps, self.event_dim))

        # If episodic, reset latest state to random initial condition
        if self.is_episodic or self.latest_state is None:
            self.reset_env()

        for environment_step_num in range(self.environment_steps):
            # Convert state to a cuda-friendly version
            cuda_state = torch.FloatTensor(self.latest_state).unsqueeze(0).to(device) # Row vector (1, state_dim)

            # Get action from policy model
            action = self.agent.act(cuda_state) # Get scaled action from policy network
            numpy_action = action.numpy() # convert to numpy array

            if self.is_gym_env:
                # Get next state, reward, and more from true dynamics
                # next_state, reward, self.has_completed, __ = self.true_dynamics.step(int(numpy_action[0,0])) # Cartpole
                next_state, reward, self.has_completed, __ = self.true_dynamics.step(numpy_action[0]) # BipedalWalker
                if self.render_env:
                    self.true_dynamics.render()
            else:
                numpy_state = cuda_state.numpy().T  # Column vector (state_dim, 1)

                # Get reward from state-action pair
                reward = self.reward(numpy_state, numpy_action) # Compute reward

                # Get next state from true dynamics
                next_state = self.true_dynamics(numpy_state, numpy_action)[:, 0] # Column vector (state_dim, 1)

            # Populate the event array with state, action, reward, and next state
            events[environment_step_num, :self.x_dim] = self.latest_state # Current state
            events[environment_step_num, self.x_dim:self.xu_dim] = action # Action from policy
            events[environment_step_num, self.xu_dim] = reward # Reward from state-action pair
            events[environment_step_num, self.xu_dim+1:self.event_dim-1] = next_state # Next state
            events[environment_step_num, self.event_dim-1] = self.has_completed # Done

            if remember:
                # Add to replay buffer
                self.agent.memorize(events[environment_step_num])

            # Update state
            self.latest_state = next_state

            # If done condition is met, break out of loop
            if self.has_completed:
                break

        # Parse out all empty rows
        events = events[:environment_step_num]

        if learn:
            for gradient_step_num in range(self.gradient_steps):
                self.agent.learn()

        return events

    def train_agent(
        self,
        num_training_iterations,
        initialization=True
    ):
        # Track rewards per iteration
        min_reward_per_iteration = 10e10
        max_reward_per_iteration = -10e10
        total_rewards_per_iteration = []

        # Run some number of initial steps using random agent
        if initialization:
            self.initialization()

        # Refer to Soft Actor-Critic algorithm (Algorithm 1) in paper
        for training_episode_num in range(num_training_iterations):
            # Take some number of steps in the environment and store the event data
            # This function will also compute gradient updates if there is enough data to start learning
            events = self.interaction(
                learn = len(self.agent.memory.data) >= self.batch_size
            )

            # Parse rewards from the events data
            rewards = events[:, self.xu_dim] # (self.environment_steps,)
            dones = torch.BoolTensor(events[:, self.event_dim-1]) # (self.environment_steps,)
            rewards[dones] = 0.0

            # Compute min, max, and mean rewards from the training episode
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            total_reward = rewards.sum()
            mean_reward = total_reward / len(rewards)

            # Update rewards data
            # if min_reward < min_reward_per_iteration:
            #     min_reward_per_iteration = min_reward
            # if max_reward > max_reward_per_iteration:
            #     max_reward_per_iteration = max_reward
            if total_reward < min_reward_per_iteration:
                min_reward_per_iteration = total_reward
            if total_reward > max_reward_per_iteration:
                max_reward_per_iteration = total_reward

            # Save total reward for the iteration
            total_rewards_per_iteration.append(total_reward)

            # Print out reward details for the episode
            print(
                "Finished episode %i, min reward = %.4f, max reward = %.4f, average reward per step = %.4f, total reward = %.4f\nmin total reward across iterations = %.4f, max total reward across iterations = %.4f, average total reward over the iterations = %.4f\n" %
                ((training_episode_num+1), min_reward, max_reward, mean_reward, total_reward, min_reward_per_iteration, max_reward_per_iteration, np.mean(total_rewards_per_iteration))
            )
        print()

        # Scatter plot of reward earned per iteration
        plt.title("Total reward per iteration")
        plt.xlabel("Iteration number")
        plt.ylabel("Total reward")
        plt.scatter(np.arange(num_training_iterations), total_rewards_per_iteration)
        plt.show()

        # Line plot of reward earned per iteration
        plt.title("Total reward per iteration")
        plt.xlabel("Iteration number")
        plt.ylabel("Total reward")
        plt.plot(total_rewards_per_iteration)
        plt.show()