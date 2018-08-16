import puzzle
from puzzle import GameGrid

import torch
import torch.optim as optim

from utils.dqn import DQN
from utils import device, ReplayMemory


if __name__ == '__main__':

    policy, target = DQN(4).to(device), DQN(4).to(device)

    try:
        policy.load_state_dict(torch.load('my_policy.pt'))
        target.load_state_dict(torch.load('my_target.pt'))
    except:
        print('Exception Raised: Files not found...')

    rm = ReplayMemory(40960)
    optimizer = optim.RMSprop(policy.parameters(), eps=1e-5)

    try:
        gamegrid = GameGrid(rm, policy, target, optimizer)
    except KeyboardInterrupt:
        print('\nKeyboard Interrupt!!!')
        try:
            print('Saving...')
            torch.save(policy.state_dict(), 'my_policy.pt')
            torch.save(target.state_dict(), 'my_target.pt')
        except Exception as e:
            print('Error :{}'.format(e))
