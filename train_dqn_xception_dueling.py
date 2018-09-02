import argparse
import gym
import gym_2048
import pickle

from play_2048 import Play2048

import torch
import torch.optim as optim

from utils import device, ReplayMemory, XceptionLikeDuelingDQN


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--epsilon',              default=1.0, type=float)
parser.add_argument('--min_epsilon',          default=0.05, type=float)
parser.add_argument('--eps_decay_rate',       default=1e-5, type=float)
parser.add_argument('--update_every',         default=100, type=int)
parser.add_argument('--n_train',              default=1200, type=int)
parser.add_argument('--batch_size',           default=512, type=int)
parser.add_argument('--gamma',                default=0.999, type=float)
parser.add_argument('--replay_memory_length', default=3000000, type=int)
parser.add_argument('--learning_rate',        default=3e-6, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])
parser.add_argument('--replay_memory',        default='replay_memory.p', type=str)
parser.add_argument('--policy_weights',       default='my_xception_policy.pt', type=str)
parser.add_argument('--target_weights',       default='my_xception_target.pt', type=str)

args = parser.parse_args()

if __name__ == '__main__':

    policy, target = XceptionLikeDuelingDQN(4).to(device), XceptionLikeDuelingDQN(4).to(device)

    try:
        policy.load_state_dict(torch.load(args.policy_weights))
        target.load_state_dict(torch.load(args.target_weights))
    except FileNotFoundError:
        print('--- Exception Raised: Files not found...')

    try:
        rm = pickle.load(open(args.replay_memory, 'rb'))
    except FileNotFoundError:
        rm = ReplayMemory(args.replay_memory_length)

    optimizer = optim.RMSprop(policy.parameters(), eps=args.learning_rate)

    env = gym.make('game-2048-v0')
    player = Play2048(env, rm, policy, target, optimizer,
                      args.batch_size, args.epsilon,
                      args.eps_decay_rate, args.min_epsilon,
                      args.n_train, args.update_every, args.gamma)

    try:
        player.play_2048(args.mode)
        if args.mode == 'train':
            print('Saving...')
            torch.save(policy.state_dict(), args.policy_weights)
            torch.save(target.state_dict(), args.target_weights)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
    finally:
        try:
            if args.mode == 'train':
                print('Saving...')
                torch.save(policy.state_dict(), args.policy_weights)
                torch.save(target.state_dict(), args.target_weights)
        except Exception as e:
            print('Error :{}'.format(e))
