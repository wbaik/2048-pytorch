import pickle
import gym
import gym_2048

from utils import ReplayMemory
from game_2048_cpp import generate_replay_memory

MAX_LENGTH = 50000000
MAX_MULTIPLIER = 4
NUM_EPISODE = 50

if __name__ == '__main__':
    env = gym.make('game-2048-v0')

    try:
        print('--- Loading ReplayMemory...')
        rm = pickle.load(open('replay_memory.p', 'rb'))

    except (FileNotFoundError, OSError):
        print('--- Pickle file not found: ReplayMemory generated...')
        rm = ReplayMemory(MAX_LENGTH)

    try:
        for multiplier in range(1, MAX_MULTIPLIER + 1):
            # Reduce the number of episodes when increasing cutoff values
            generate_replay_memory(rm,
                                   env,
                                   1024 * (1 << multiplier),
                                   NUM_EPISODE // multiplier)

    except KeyboardInterrupt:
        print('--- Exiting from Keyboard Interrupt')
        print('--- Saving Replay Memory to pickle')
        pickle.dump(rm, open('replay_memory.p', 'wb'))
