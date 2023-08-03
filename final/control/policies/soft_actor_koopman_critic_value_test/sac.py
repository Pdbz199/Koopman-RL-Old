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

        # self.critic = QNetwork(state_dim, action_dim, args.hidden_size).to(device=self.device)
        self.critic = KoopmanQNetwork(koopman_tensor).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # self.critic_target = QNetwork(state_dim, action_dim, args.hidden_size).to(device=self.device)
        self.critic_target = KoopmanQNetwork(koopman_tensor).to(device=self.device)
        hard_update(self.critic_target, self.critic)

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
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, state_prime_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        state_prime_batch = torch.FloatTensor(state_prime_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            action_prime_batch, action_prime_log_prob, _ = self.policy.sample(state_prime_batch)

            # phi_state_prime_batch = self.phi(next_state_batch.T).T # phi(x')
            # psi_action_prime_batch = self.psi(next_state_action.T).T # psi(u')

            # qf1_next_target, qf2_next_target = self.critic_target(phi_state_prime_batch, psi_action_prime_batch)
            qf1_next_target, qf2_next_target = self.critic_target(state_prime_batch, action_prime_batch)
            # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha*next_state_log_pi
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha*action_prime_log_prob
            next_q_value = reward_batch + mask_batch * self.gamma*min_qf_next_target

        # phi_state_batch = self.phi(state_batch.T).T
        # psi_action_batch = self.psi(action_batch.T).T
        # qf1, qf2 = self.critic(phi_state_batch, psi_action_batch) # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch) # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        # qf1_pi, qf2_pi = self.critic(phi_state_batch, self.psi(pi.T).T)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

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

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()