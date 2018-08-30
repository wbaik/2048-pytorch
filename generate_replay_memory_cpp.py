import pickle
import gym
import gym_2048

from utils import ReplayMemory
from game_2048_cpp import generate_replay_memory


if __name__ == '__main__':
    env = gym.make('game-2048-v0')

    try:
        print('--- Loading ReplayMemory...')
        rm = pickle.load(open('replay_memory.p', 'rb'))

    except (FileNotFoundError, OSError):
        print('--- Pickle file not found: ReplayMemory generated...')
        rm = ReplayMemory(3000000)

    generate_replay_memory(rm, env, 2048)

    print('--- Saving Replay Memory to pickle')
    pickle.dump(rm, open('replay_memory.p', 'wb'))
