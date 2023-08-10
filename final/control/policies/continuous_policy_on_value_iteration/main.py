import argparse
import datetime
import gym
import itertools
import numpy as np
import pickle
import sys
import torch
import torch.nn as nn

from load_discrete_value_iteration_policy import load_koopman_value_iteration_policy
from models import GaussianPolicy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../')
from linear_system.dynamics_env import LinearSystem
from lorenz.dynamics_env import Lorenz
from fluid_flow.dynamics_env import FluidFlow
from double_well.dynamics_env import DoubleWell

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Continuous Value Iteration Policy Args')
parser.add_argument('--env_name', default="LinearSystem-v0",
                    help='Gym environment (default: LinearSystem-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--eval_frequency', type=int, default=10,
                    help='Number of iterations to run between each evaluation step (default: 10)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Extract system name
env_name_to_system_name = {
    "LinearSystem-v0": "linear_system",
    "FluidFlow-v0": "fluid_flow",
    "Lorenz-v0": "lorenz",
    "DoubleWell-v0": "double_well"
}
system_name = env_name_to_system_name[args.env_name]

# Decide which device to use
device = torch.device("cuda" if args.cuda else "cpu")

# Environment
training_env = gym.make(args.env_name)
cvi_env = gym.make(args.env_name)
# lqr_env = gym.make(args.env_name)

state_dim = training_env.observation_space.shape[0]
action_dim = training_env.action_space.shape[0]
action_space = training_env.action_space

# Set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Append to sys path for loading koopman tensor and LQR policy
sys.path.append('../../../../')

# Load LQR policy
with open(f'../../{system_name}/analysis/tmp/lqr/policy.pickle', 'rb') as handle:
    lqr_policy = pickle.load(handle)

# Load Koopman tensor with pickle
with open(f'../../{system_name}/analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    koopman_tensor = pickle.load(handle)

sys.path.append(f'../../{system_name}/')
from dynamics import all_actions, f

# Load Koopman value iteration policy
koopman_value_iteration_policy = load_koopman_value_iteration_policy(
    f,
    koopman_tensor,
    all_actions,
    training_env.cost_fn,
    args.seed,
    system_name=system_name
)
print(f"Value function weights:\n{koopman_value_iteration_policy.value_function_weights}")

# Continuous policy
if args.policy == "Gaussian":
    # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
    if args.automatic_entropy_tuning is True:
        target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = Adam([log_alpha], lr=args.lr)

    policy = GaussianPolicy(
        state_dim,
        action_dim,
        args.hidden_size,
        action_space
    ).to(device=device)
    policy_optim = Adam(policy.parameters(), lr=args.lr)

kl_loss = nn.KLDivLoss(reduction="batchmean")
# kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

def select_action(state, evaluate=False):
    state = torch.FloatTensor(state).to(device=device).unsqueeze(0)
    if evaluate is False:
        action, _, _ = policy.sample(state)
    else:
        _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]

# Tensorboard
writer = SummaryWriter(
    'runs/{}_CVI_{}_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else ""
    )
)

# Training Loop
total_numsteps = 0
updates = 0
eval_steps = 0

for iteration_num in itertools.count(1):
    # Get random sample of xs and phi(x)s from dataset
    state_batch_indices = np.random.choice(
        koopman_tensor.X.shape[1],
        args.batch_size,
        replace=False
    )
    state_batch = torch.Tensor(koopman_tensor.X[:, state_batch_indices].T) # (batch_size, state_dim)

    # Get optimal actions from value iteration's softmax policy
    optimal_log_prob_batch = np.zeros((args.batch_size, 1))
    for i in range(args.batch_size):
        _, log_prob_i = koopman_value_iteration_policy.get_action_and_log_prob(np.vstack(state_batch[i]))
        optimal_log_prob_batch[i] = log_prob_i[0]
    optimal_log_prob_batch = torch.Tensor(optimal_log_prob_batch)

    # Sample batch of actions from current policy
    _, policy_log_prob_batch, _ = policy.sample(state_batch) # returns action, log prob, mean

    # Number of updates per step in environment
    for i in range(args.updates_per_step):
        # Update parameters of all the networks
        # SAC policy loss: policy_loss = ((self.alpha * log_pi) - soft_quality).mean()
        policy_loss = (args.alpha*policy_log_prob_batch - optimal_log_prob_batch).mean()
        print(f"Policy loss: {policy_loss.item()}")
        # policy_loss = (args.alpha*policy_log_prob_batch - optimal_log_prob_batch.exp()).mean()
        # print(f"Policy loss: {policy_loss.item()}")
        # policy_loss = kl_loss(optimal_log_prob_batch, (args.alpha*policy_log_prob_batch).exp())
        # print(f"Policy loss: {policy_loss.item()}")

        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
        # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
        # writer.add_scalar('loss/policy', policy_loss, updates)
        # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        updates += 1

    total_numsteps += 1

    print("Iteration: {}, total numsteps: {}".format(iteration_num, total_numsteps))

    if iteration_num % args.eval_frequency == 0:
        cvi_avg_reward = None
        episodes = None

        if args.eval is True:
            cvi_avg_reward = 0
            # lqr_avg_reward = 0
            # episodes = 200
            episodes = 100
            # episodes = 1

            for _  in range(episodes):
                # initial_state = np.array([1, 1, 1])

                # cvi_env.reset(options={"state": initial_state})
                cvi_env.reset()
                cvi_state = cvi_env.state
                cvi_episode_reward = 0

                # lqr_env.reset(options={"state": initial_state})
                # lqr_env.reset(options={"state": cvi_state})
                # lqr_state = lqr_env.state
                # lqr_episode_reward = 0

                done = False

                while not done:
                    cvi_action = select_action(cvi_state, evaluate=True)
                    # cvi_action = select_action(cvi_state, evaluate=False)
                    # lqr_action = lqr_policy.get_action(np.vstack(lqr_state))[0]

                    cvi_state, cvi_reward, done, _, __ = cvi_env.step(cvi_action)
                    # cvi_env.render()
                    cvi_episode_reward += cvi_reward

                    # lqr_state, lqr_reward, done, _, __ = lqr_env.step(lqr_action)
                    # lqr_env.render()
                    # lqr_episode_reward += lqr_reward

                # print("cvi Reward:", cvi_episode_reward)
                # print("LQR Reward", lqr_episode_reward, "\n")

                cvi_avg_reward += cvi_episode_reward
                # lqr_avg_reward += lqr_episode_reward
            cvi_avg_reward /= episodes
            # lqr_avg_reward /= episodes

            # print("cvi Average Reward:", cvi_avg_reward)
            # print("LQR Average Reward:", lqr_avg_reward, "\n")

            writer.add_scalar('avg_reward/test', cvi_avg_reward, eval_steps)
            eval_steps += 1

        if cvi_avg_reward is not None:
            rounded_cvi_avg_reward = round(cvi_avg_reward, 2)
        else:
            rounded_cvi_avg_reward = None

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, rounded_cvi_avg_reward))
        print("----------------------------------------")

    # break

training_env.close()