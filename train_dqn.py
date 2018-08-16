import argparse

import puzzle
from puzzle import GameGrid

import torch
import torch.optim as optim

from utils.dqn import DQN
from utils import device, ReplayMemory


parser = argparse.ArgumentParser(description='Hyper-parameters for the DQN training')
parser.add_argument('--epsilon', default=1.0, type=float)
parser.add_argument('--min_epsilon', default=0.2, type=float)
parser.add_argument('--eps_decay_rate', default=1e-4, type=float)
parser.add_argument('--update_every', default=40, type=int)
parser.add_argument('--n_train', default=5000, type=int)
parser.add_argument('--batch_size', default=2*1024, type=int)
parser.add_argument('--gamma', default=0.999, type=float)
parser.add_argument('--replay_memory_length', default=40960, type=int)

args = parser.parse_args()

if __name__ == '__main__':

    policy, target = DQN(4).to(device), DQN(4).to(device)

    try:
        policy.load_state_dict(torch.load('my_policy.pt'))
        target.load_state_dict(torch.load('my_target.pt'))
    except:
        print('Exception Raised: Files not found...')

    rm = ReplayMemory(args.replay_memory_length)
    optimizer = optim.RMSprop(policy.parameters(), eps=1e-5)

    try:
        gamegrid = GameGrid(rm, policy, target, optimizer,
                            args.epsilon, args.min_epsilon, args.eps_decay_rate,
                            args.update_every, args.n_train,
                            args.batch_size, args.gamma)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
        try:
            print('Saving...')
            torch.save(policy.state_dict(), 'my_policy.pt')
            torch.save(target.state_dict(), 'my_target.pt')
        except Exception as e:
            print('Error :{}'.format(e))
