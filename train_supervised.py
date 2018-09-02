import argparse
import gym
import gym_2048
import pickle

from play_2048 import Play2048

import torch
import torch.optim as optim
from tqdm import tqdm

from utils import device, ReplayMemory, SupervisedModel, train_supervised


parser = argparse.ArgumentParser(description='Hyper-parameters for the training')
parser.add_argument('--n_epoch',              default=200000, type=int)
parser.add_argument('--batch_size',           default=128, type=int)
parser.add_argument('--replay_memory_length', default=5000000, type=int)
parser.add_argument('--learning_rate',        default=3e-6, type=float)
parser.add_argument('--mode',                 default='train', type=str, choices=['train', 'test'])
parser.add_argument('--replay_memory',        default='replay_memory.p', type=str)
parser.add_argument('--weights',              default='supervised_model.pt', type=str)

args = parser.parse_args()

if __name__ == '__main__':

    model = SupervisedModel(4).to(device)

    try:
        model.load_state_dict(torch.load(args.weights))
    except FileNotFoundError:
        print('--- Exception Raised: Files not found...')

    try:
        rm = pickle.load(open(args.replay_memory, 'rb'))
    except FileNotFoundError:
        rm = ReplayMemory(args.replay_memory_length)

    optimizer = optim.RMSprop(model.parameters(), eps=args.learning_rate)

    if args.mode == 'train':
        try:
            for _ in tqdm(range(args.n_epoch)):
                train_supervised(model, rm, args.batch_size, optimizer)
        except KeyboardInterrupt:
            print('\nKeyboard Interrupt!!!')
        finally:
            print('Saving...')
            torch.save(model.state_dict(), args.weights)

    else:
        Play2048.supervised_model_test(model, 10000, 100)
