import numpy as np
import os
import pickle
import sys
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

# Load LQR policy
sys.path.append('../../../../')
system_name = "linear_system"
# system_name = "fluid_flow"
# system_name = "lorenz"
# system_name = "double_well"
with open(f'../../{system_name}/analysis/tmp/lqr/policy.pickle', 'rb') as handle:
    lqr_policy = pickle.load(handle)

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
        self.critic = KoopmanQFunction(koopman_tensor, self.gamma)
        self.critic_target = KoopmanQFunction(koopman_tensor, self.gamma)
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

    def update_critic_weights(self, x_batch, u_batch, r_batch, x_prime_batch):
        """
            Update the weights for the value function in the dictionary space.
        """

        with torch.no_grad():
            # Get batch size from input
            batch_size = x_batch.shape[0]

            # Get actions and log probabiltiies from current policy
            u_prime_batch, log_prob_prime_batch, _ = self.policy.sample(x_prime_batch)

            # Prepare batches for numpy functions
            numpy_x_batch = x_batch.numpy().T # (state_dim, batch_size)
            numpy_u_batch = u_batch.numpy().T # (action_dim, batch_size)
            numpy_r_batch = r_batch.numpy().T # (1, batch_size)
            numpy_x_prime_batch = x_prime_batch.numpy().T # (state_dim, batch_size)

            numpy_u_prime_batch = u_prime_batch.numpy().T # (1, batch_size)
            numpy_log_prob_prime_batch = log_prob_prime_batch.numpy().T # (1, batch_size)

            # Compute rewards and expected phi(x')s
            r_prime_batch = torch.zeros((batch_size, 1))
            # expected_phi_x_prime_batch = np.zeros((self.koopman_tensor.Phi_X.shape[0], batch_size))
            for i in range(batch_size):
                r_prime_batch[i, 0] = self.env.reward(
                    np.vstack(numpy_x_prime_batch[:, i]),
                    np.vstack(numpy_u_prime_batch[:, i])
                )[0, 0]

                # expected_phi_x_prime_batch[:, i] = self.koopman_tensor.phi_f(
                #     np.vstack(numpy_x_batch[:, i]),
                #     np.vstack(numpy_u_batch[:, i])
                # )[:, 0]
            # normalized_r_prime_batch = (r_prime_batch - r_prime_batch.mean()) / (r_prime_batch.std() + epsilon)
            phi_x_batch = self.koopman_tensor.phi(numpy_x_batch) # (phi_dim, batch_size)

            # Compute target Q(s', a')
            target_Q_x_u_prime_batch = self.critic_target(r_prime_batch, x_prime_batch, u_prime_batch).numpy() - \
                                    self.alpha*numpy_log_prob_prime_batch # (1, batch_size)
            # target_Q_x_u_prime_batch = self.critic_target(normalized_r_prime_batch, x_prime_batch, u_prime_batch).numpy() - \
            #                         self.alpha*numpy_log_prob_prime_batch # (1, batch_size)
            Q_x_u_prime_batch = numpy_r_batch + self.gamma*target_Q_x_u_prime_batch # (1, batch_size)

            # Update value function weights
            self.critic.w = torch.linalg.lstsq(
                # torch.Tensor(expected_phi_x_prime_batch.T),
                torch.Tensor(phi_x_batch.T),
                torch.Tensor((Q_x_u_prime_batch - numpy_r_batch).T)
            ).solution

            # norms = np.linalg.norm(
            #     (Q_x_u_prime_batch - numpy_r_batch) - (self.critic.w.numpy().T @ expected_phi_x_prime_batch),
            #     axis=0
            # ) / np.linalg.norm(
            #     Q_x_u_prime_batch - numpy_r_batch,
            #     axis=0
            # ).mean()
            # print(norms.mean())

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, log_prob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # Convert data to usable tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        log_prob_batch = torch.FloatTensor(log_prob_batch).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Normalize batches
        # normalized_reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + epsilon)

        """ UPDATE CRITIC """

        # OLS to update weight vector for critic
        self.update_critic_weights(state_batch, action_batch, reward_batch, next_state_batch)
        # self.update_critic_weights(state_batch, action_batch, normalized_reward_batch, next_state_batch)

        """ UPDATE POLICY """

        # Sample actions and log probabilities from latest policy
        new_action_batch, new_log_prob_batch, _ = self.policy.sample(state_batch)

        # Compute Q_π(s, a)
        Q_pi_batch = self.critic(reward_batch, state_batch, new_action_batch).T
        # Q_pi_batch = self.critic(normalized_reward_batch, state_batch, new_action_batch).T

        # Equation 12 in SAC Paper
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha*new_log_prob_batch) - Q_pi_batch).mean()

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