import pickle
import gym
import gym_2048

import numpy as np

from itertools import count

from utils import ReplayMemory, direction


env = gym.make('game-2048-v0')

KEY_TO_VAL = {
    'left':     3,
    'right':    1,
    'up':       0,
    'down':     2
}

if __name__ == '__main__':

    try:
        print('--- Loading ReplayMemory...')
        rm = pickle.load(open('replay_memory.p', 'rb'))

    except (FileNotFoundError, OSError):
        print('--- Pickle file not found: ReplayMemory generated...')
        rm = ReplayMemory(1000000)

    try:
        print('--- Generating data... Current length of the rm : {}'.format(rm.__len__()))

        while True:
            state = env.reset()

            for t in count(1):

                key = direction(state)
                action = KEY_TO_VAL[key]

                next_state, reward, done, info = env.step(action)
                this_state_max = np.max(next_state)

                rm.push(state, action, next_state, reward)
                # To make replay_memory have enough variations...
                # if this_state_max >= 1024.0:
                #     rm.push(state, action, next_state, reward)
                # else:
                #     if np.random.random() < 0.3:
                #         rm.push(state, action, next_state, reward)

                if done:
                    print('Current t : {}'.format(t))
                    env.render('human')
                    print('--- Generating data... Current length of the rm : {}'.format(rm.__len__()))
                    break

                state = next_state

    except KeyboardInterrupt:
        print('--- Exiting from Keyboard Interrupt')

    finally:
        print('--- Saving Replay Memory to pickle')
        pickle.dump(rm, open('replay_memory.p', 'wb'))




