import os
import sys
import torch
import torch.nn.functional as F

from model import (
    DeterministicPolicy,
    GaussianPolicy,
    KoopmanQNetwork,
    KoopmanVNetwork,
    QNetwork,
)
from scipy.special import comb
from torch.optim import Adam
from utils import soft_update, hard_update

sys.path.append('../../../')
# import observables_pytorch as observables
import observables

class SAC(object):
    def __init__(self, env, args, koopman_tensor):

        self.env = env

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        state_monomials_order = 2
        action_monomials_order = 2
        self.phi = observables.monomials(state_monomials_order)
        self.psi = observables.monomials(action_monomials_order)
        phi_dim = int( comb( state_monomials_order+state_dim, state_monomials_order ) )
        psi_dim = int( comb( action_monomials_order+action_dim, action_monomials_order ) )
        action_space = env.action_space

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.soft_value = KoopmanVNetwork(koopman_tensor).to(device=self.device)
        self.soft_value_optim = Adam(self.soft_value.parameters(), lr=args.lr)

        self.soft_value_target = KoopmanVNetwork(koopman_tensor).to(device=self.device)
        hard_update(self.soft_value_target, self.soft_value)

        self.soft_quality = KoopmanQNetwork(koopman_tensor).to(device=self.device)
        self.soft_quality_optim = Adam(self.soft_quality.parameters(), lr=args.lr)

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

    def update_parameters(self, memory, batch_size, updates):
        """ Sample a batch from replay buffer """

        state_batch, action_batch, reward_batch, state_prime_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        state_prime_batch = torch.FloatTensor(state_prime_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        """ Update soft value function """

        with torch.no_grad():
            _, current_policy_action_log_prob_batch, __ = self.policy.sample(state_batch)

        predicted_soft_value = self.soft_value(state_batch)
        true_soft_value = self.soft_quality(state_batch, action_batch) - current_policy_action_log_prob_batch
        soft_value_loss = F.mse_loss(predicted_soft_value, true_soft_value)

        self.soft_value_optim.zero_grad()
        soft_value_loss.backward()
        self.soft_value_optim.step()

        """ Update soft quality function """

        with torch.no_grad():
            target_quality = reward_batch + mask_batch * self.gamma*self.soft_value_target(state_prime_batch)

        soft_quality = self.soft_quality(state_batch, action_batch)
        soft_quality_loss = F.mse_loss(soft_quality, target_quality) # JQ = 𝔼(st,at)~D[0.5(Q(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.soft_quality_optim.zero_grad()
        soft_quality_loss.backward()
        self.soft_quality_optim.step()

        """ Update policy """

        pi, log_pi, _ = self.policy.sample(state_batch)
        soft_quality = self.soft_quality(state_batch, pi)

        policy_loss = ((self.alpha * log_pi) - soft_quality).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        """ Update target parameters """

        if updates % self.target_update_interval == 0:
            soft_update(self.soft_value_target, self.soft_value, self.tau)

        """ Update entropy variable if enabled """

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

        return soft_value_loss.item(), soft_quality_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({
            'soft_value_state_dict': self.soft_value.state_dict(),
            'soft_value_optimizer_state_dict': self.soft_value_optim.state_dict(),
            'soft_quality_state_dict': self.soft_quality.state_dict(),
            'soft_quality_optimizer_state_dict': self.soft_quality_optim.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'soft_value_target_state_dict': self.soft_value_target.state_dict()
        }, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.soft_value.load_state_dict(checkpoint['soft_value_state_dict'])
            self.soft_value_optim.load_state_dict(checkpoint['soft_value_optimizer_state_dict'])
            self.soft_quality.load_state_dict(checkpoint['soft_quality_state_dict'])
            self.soft_quality_optim.load_state_dict(checkpoint['soft_quality_optimizer_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.soft_value_target.load_state_dict(checkpoint['soft_value_target_state_dict'])

            if evaluate:
                self.policy.eval()
                self.soft_value.eval()
                self.soft_value_target.eval()
                self.soft_quality.eval()
            else:
                self.policy.train()
                self.soft_value.train()
                self.soft_value_target.train()
                self.soft_quality.train()