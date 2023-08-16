"""
cd ./final/control/policies/soft_actor_koopman_critic_test
python main.py --start_steps=0 --alpha=1.0 --batch_size=256 --eval_frequency=10 --eval_episodes=100 --env_name=FluidFlow-v0 --use_neural_networks=True
"""

import argparse
import datetime
import gym
import itertools
import numpy as np
import pickle
import sys
import torch

from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

sys.path.append('../../')
from linear_system.dynamics_env import LinearSystem
from lorenz.dynamics_env import Lorenz
from fluid_flow.dynamics_env import FluidFlow
from double_well.dynamics_env import DoubleWell

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="LinearSystem-v0",
                    help='Gym environment (default: LinearSystem-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--eval_frequency', type=int, default=10,
                    help='Number of iterations to run between each evaluation step (default: 10)')
parser.add_argument('--eval_episodes', type=int, default=100,
                    help='Number of episodes to perform per evaluation step (default: 100)')
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
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
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
parser.add_argument('--use_neural_networks', type=bool, default=False,
                    help='Use neural networks for all networks in SAC algorithm (default: False)')
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
try:
    system_name = env_name_to_system_name[args.env_name]
except:
    system_name = "linear_system"

# Environment
# env = NormalizedActions(gym.make(args.env_name))
training_env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)
sac_env = gym.make(args.env_name)
# sac_env.action_space.seed(args.seed)
lqr_env = gym.make(args.env_name)
# lqr_env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Append to sys path for loading tensor and LQR policy
sys.path.append('../../../../')

# Load LQR policy
with open(f'../../{system_name}/analysis/tmp/lqr/policy.pickle', 'rb') as handle:
    lqr_policy = pickle.load(handle)

# Load Koopman tensor with pickle
with open(f'../../{system_name}/analysis/tmp/path_based_tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

# Agent
agent = SAC(training_env, args, tensor)
# agent.load_checkpoint(ckpt_path=f"checkpoints/sac_checkpoint_{args.env_name}_")

# Tensorboard
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else "no_autotune"
    )
)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
eval_steps = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state, _ = sac_env.reset()
    # done = True

    while not done:
        if args.start_steps > total_numsteps:
            action = sac_env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (
                    soft_value_loss,
                    soft_quality_1_loss,
                    soft_quality_2_loss,
                    policy_loss,
                    entropy_loss,
                    alpha
                ) = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/soft_value', soft_value_loss, updates)
                writer.add_scalar('loss/soft_quality_1', soft_quality_1_loss, updates)
                writer.add_scalar('loss/soft_quality_2', soft_quality_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', entropy_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _, __ = sac_env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == sac_env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    # if total_numsteps > args.num_steps:
    #     break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print(
        "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode,
            total_numsteps,
            episode_steps,
            round(episode_reward, 2)
        )
    )

    if i_episode % args.eval_frequency == 0:
        if len(memory) > args.batch_size:
            agent.save_checkpoint(args.env_name)

        sac_avg_reward = None
        num_eval_episodes = None

        if args.eval is True and len(memory) > args.batch_size:
            sac_avg_reward = 0
            lqr_avg_reward = 0
            num_eval_episodes = args.eval_episodes

            for _  in range(num_eval_episodes):
                # initial_state = np.array([1, 1, 1])

                # sac_env.reset(options={"state": initial_state})
                sac_env.reset()
                sac_state = sac_env.state
                sac_episode_reward = 0

                # lqr_env.reset(options={"state": initial_state})
                lqr_env.reset(options={"state": sac_state})
                lqr_state = lqr_env.state
                lqr_episode_reward = 0

                done = False

                while not done:
                    # sac_action = agent.select_action(sac_state, evaluate=True)
                    sac_action = agent.select_action(sac_state, evaluate=False)
                    lqr_action = lqr_policy.get_action(np.vstack(lqr_state))[0]

                    sac_state, sac_reward, done, _, __ = sac_env.step(sac_action)
                    # sac_env.render()
                    sac_episode_reward += sac_reward

                    lqr_state, lqr_reward, done, _, __ = lqr_env.step(lqr_action)
                    # lqr_env.render()
                    lqr_episode_reward += lqr_reward

                # print("SAC Reward:", sac_episode_reward)
                # print("LQR Reward", lqr_episode_reward, "\n")

                sac_avg_reward += sac_episode_reward
                lqr_avg_reward += lqr_episode_reward
            sac_avg_reward /= num_eval_episodes
            lqr_avg_reward /= num_eval_episodes

            print("SAC Average Reward:", sac_avg_reward)
            print("LQR Average Reward:", lqr_avg_reward, "\n")

            writer.add_scalar('avg_reward/test', sac_avg_reward, eval_steps)
            eval_steps += 1

        if sac_avg_reward is not None:
            rounded_sac_avg_reward = round(sac_avg_reward, 2)
        else:
            rounded_sac_avg_reward = None

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(num_eval_episodes, rounded_sac_avg_reward))
        print("----------------------------------------")

    # break

training_env.close()
sac_env.close()
lqr_env.close()