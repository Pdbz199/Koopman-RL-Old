import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from utils import soft_update, hard_update
from model import (
    DeterministicPolicy,
    GaussianPolicy,
    KoopmanQFunction,
    QNetwork
)

epsilon = np.finfo(np.float64).eps

class SAKC(object):
    def __init__(self, env, args, koopman_tensor):
        self.env = env

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space
        step_size = 0.1
        self.all_actions = np.array([
            np.arange(
                action_space.low[0],
                action_space.high[0]+step_size,
                step=step_size
            )
        ])

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.regularization_lambda = args.regularization_lambda

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.koopman_tensor = koopman_tensor

        # self.critic = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(device=self.device)
        # self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        # self.critic_target = QNetwork(state_dim, action_space.shape[0], args.hidden_size).to(device=self.device)
        # hard_update(self.critic_target, self.critic, is_koopman=False)
        self.critic = KoopmanQFunction(koopman_tensor)
        self.critic_target = KoopmanQFunction(koopman_tensor)
        hard_update(self.critic_target, self.critic, is_koopman=True)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(state_dim, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                state_dim,
                action_space.shape[0],
                args.hidden_size,
                action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, return_log_prob=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action, log_prob, _ = self.policy.sample(state) # Sample from policy
        action = action.detach().cpu().numpy()[0]

        if return_log_prob:
            log_prob = log_prob.detach().cpu().numpy()[0, 0]
            return action, log_prob

        return action

    def update_critic_weights(self, x_batch, u_batch, log_prob_batch, reward_batch):
        """
            Update the weights for the value function in the dictionary space.
        """

        # For use with numpy functions
        x_batch = x_batch.detach().numpy().T # (state_dim, batch_size)
        u_batch = u_batch.detach().numpy().T # (1, batch_size)
        log_prob_batch = log_prob_batch.detach().numpy().T # (1, batch_size)
        reward_batch = reward_batch.detach().numpy().T # (1, batch_size)

        # Batch normalization
        # x_batch = (x_batch - np.vstack(x_batch.mean(axis=1))) / (np.vstack(x_batch.std(axis=1)) + epsilon)
        # reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + epsilon)

        # Compute phi states for given batch
        phi_x_batch = self.koopman_tensor.phi(x_batch) # (phi_dim, batch_size)

        # Compute expected phi(x')
        expected_phi_x_prime_batch = np.zeros_like(phi_x_batch) # (phi_dim, batch_size)
        for i in range(phi_x_batch.shape[1]):
            K_u = self.koopman_tensor.K_(np.vstack(u_batch[:, i])) # (phi_dim, phi_dim)
            expected_phi_x_prime_batch[:, i] = (K_u @ np.vstack(phi_x_batch[:, i]))[:, 0] # (phi_dim,)

        # Compute expected V(x')s
        expected_V_x_prime_batch = self.critic.w.numpy().T @ expected_phi_x_prime_batch # (1, batch_size)

        # Update value function weights
        self.critic.w = torch.linalg.lstsq(
            torch.Tensor(phi_x_batch.T),
            torch.Tensor((
                (-reward_batch + \
                 (self.regularization_lambda * log_prob_batch) + \
                    (self.gamma * expected_V_x_prime_batch)) * \
                    np.exp(log_prob_batch)).T
            )
        ).solution / 100

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, log_prob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # Normalize the states and the rewards
        # state_batch = (state_batch - np.vstack(state_batch.mean(axis=1))) / (np.vstack(state_batch.std(axis=1)) + epsilon)
        # reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + epsilon)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        log_prob_batch = torch.FloatTensor(log_prob_batch).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        """ UPDATE CRITIC """

        # OLS to update weight vector for critic
        self.update_critic_weights(state_batch, action_batch, log_prob_batch, reward_batch)

        # with torch.no_grad():
        #     next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
        #     qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        #     min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        #     next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_loss = qf1_loss + qf2_loss

        # self.critic_optim.zero_grad()
        # qf_loss.backward()
        # self.critic_optim.step()

        """ UPDATE POLICY """

        new_action_batch, log_prob_batch, _ = self.policy.sample(state_batch)

        # Q_pi_1_batch, Q_pi_2_batch = self.critic(state_batch, new_action_batch)
        # Q_pi_batch = torch.min(Q_pi_1_batch, Q_pi_2_batch)
        Q_pi_batch = self.critic(reward_batch, state_batch, new_action_batch)

        # Equation 12 in SAC Paper
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_prob_batch) - Q_pi_batch).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        """ CRITIC TARGET UPDATE """

        # soft_update(self.critic_target, self.critic, self.tau, is_koopman=False)
        soft_update(self.critic_target, self.critic, self.tau, is_koopman=True)

        """ UPDATE ENTROPY TERM """

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob_batch + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        return policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_weights': self.critic.w,
            'critic_target_weights': self.critic_target.w,
            'policy_optimizer_state_dict': self.policy_optim.state_dict()
        }, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.w = checkpoint['critic_weights']
            self.critic_target.w = checkpoint['critic_target_weights']
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                # self.critic.eval()
                # self.critic_target.eval()
            else:
                self.policy.train()
                # self.critic.train()
                # self.critic_target.train()