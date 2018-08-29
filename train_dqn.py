import argparse
import gym
import gym_2048
import pickle

from play_2048 import Play2048

import torch
import torch.optim as optim

from utils.dqn import DQN
from utils import device, ReplayMemory


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--epsilon',              default=1.0, type=float)
parser.add_argument('--min_epsilon',          default=0.2, type=float)
parser.add_argument('--eps_decay_rate',       default=1e-6, type=float)
parser.add_argument('--update_every',         default=300, type=int)
parser.add_argument('--n_train',              default=100000, type=int)
parser.add_argument('--batch_size',           default=1024, type=int)
parser.add_argument('--gamma',                default=0.99, type=float)
parser.add_argument('--replay_memory_length', default=1000000, type=int)
parser.add_argument('--learning_rate',        default=1e-5, type=float)
parser.add_argument('--mode',                 default='train', type=str)


args = parser.parse_args()

if __name__ == '__main__':

    policy, target = DQN(4).to(device), DQN(4).to(device)

    try:
        policy.load_state_dict(torch.load('my_policy.pt'))
        target.load_state_dict(torch.load('my_target.pt'))
    except FileNotFoundError:
        print('--- Exception Raised: Files not found...')

    try:
        rm = pickle.load(open('replay_memory.p', 'rb'))
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
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
        try:
            print('Saving...')
            torch.save(policy.state_dict(), 'my_policy.pt')
            torch.save(target.state_dict(), 'my_target.pt')
        except Exception as e:
            print('Error :{}'.format(e))
