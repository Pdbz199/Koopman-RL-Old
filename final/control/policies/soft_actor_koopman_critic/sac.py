import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import (
    GaussianPolicy,
    KoopmanQFunction,
    DeterministicPolicy
)

class SAKC(object):
    def __init__(self, env, args, koopman_tensor=None):

        self.env = env

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.koopman_tensor = koopman_tensor

        self.critic = KoopmanQFunction(koopman_tensor)

        self.critic_target = KoopmanQFunction(koopman_tensor)

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

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update_critic_weights(self):
        """
            Update the weights for the value function in the dictionary space.
        """

        # TODO: CHECK SHAPES OF ALL THINGS

        # Take state sample from dataset
        x_batch_indices = np.random.choice(
            self.koopman_tensor.X.shape[1],
            self.w_hat_batch_size,
            replace=False
        )
        x_batch = self.koopman_tensor.X[:, x_batch_indices]
        phi_x_batch = self.koopman_tensor.Phi_X[:, x_batch_indices]

        # Compute policy probabilities for each state in the batch
        with torch.no_grad():
            _, log_pi_response = self.policy.sample(torch.Tensor(x_batch.T)).T.numpy()
            pi_response = np.exp(log_pi_response)

        # Compute phi_x_prime for all states in the batch using all actions in the action space
        K_us = self.koopman_tensor.K_(self.all_actions)
        phi_x_prime_batch = K_us @ phi_x_batch

        # Compute expected phi(x')
        phi_x_prime_batch_prob = np.einsum(
            'upw,uw->upw',
            phi_x_prime_batch,
            pi_response
        )
        expectation_term_1 = phi_x_prime_batch_prob.sum(axis=0)

        # Compute expected reward
        rewards_batch = -self.env.reward(x_batch, self.all_actions)
        reward_batch_prob = np.einsum(
            'uw,uw->uw',
            rewards_batch,
            pi_response
        )
        expectation_term_2 = np.array([
            reward_batch_prob.sum(axis=0)
        ])

        # Update value function weights
        self.critic.w = torch.linalg.lstsq(
            torch.Tensor((phi_x_batch - (self.discount_factor * expectation_term_1)).T),
            torch.Tensor(expectation_term_2.T)
        ).solution
        

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf_pi = self.critic(reward_batch, state_batch, pi)

        policy_loss = ((self.alpha * log_pi) - qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        # OLS to update weight vector for critic
        self.update_critic_weights()

        # Occasionally update critic target
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

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
            self.critic.w = checkpoint['critic_state_dict']
            self.critic_target = ['critic_target_state_dict']
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                # self.critic.eval()
                # self.critic_target.eval()
            else:
                self.policy.train()
                # self.critic.train()
                # self.critic_target.train()