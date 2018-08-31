#!/usr/bin/python
# -*- coding: utf-8 -*-

''' Help the user achieve a high score in a real game of 2048 by using a move searcher. '''

from __future__ import print_function
import ctypes
import logging
import pickle
import os

import numpy as np
from itertools import count


FILE_NAME = 'generating.log'

try:
    os.remove(FILE_NAME)
except FileNotFoundError:
    print('File Not Found')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler(FILE_NAME)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Enable multithreading?
MULTITHREAD = False

for suffix in ['so', 'dll', 'dylib']:
    dllfn = 'game_2048_cpp/bin/2048.' + suffix
    if not os.path.isfile(dllfn):
        continue
    ailib = ctypes.CDLL(dllfn)
    break
else:
    print("Couldn't find 2048 library bin/2048.{so,dll,dylib}! Make sure to build it first.")
    exit()

ailib.init_tables()
ailib.find_best_move.argtypes = [ctypes.c_uint64]
ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
ailib.score_toplevel_move.restype = ctypes.c_float


def to_c_board(m):
    board = 0
    i = 0
    for row in m:
        for c in row:
            board |= int(c) << (4*i)
            i += 1
    return board


def print_board(m):
    for row in m:
        for c in row:
            print('%8d' % c, end=' ')
        print()


def _to_val(c):
    if c == 0:
        return 0
    return 1 << c


def to_val(m):
    return [[_to_val(c) for c in row] for row in m]


def _to_score(c):
    if c <= 1:
        return 0
    return (c - 1) * (1 << c)


def to_score(m):
    return [[_to_score(c) for c in row] for row in m]


if MULTITHREAD:
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(4)

    def score_toplevel_move(args):
        return ailib.score_toplevel_move(*args)

    def find_best_move(m):
        board = to_c_board(m)

        print_board(to_val(m))

        scores = pool.map(score_toplevel_move, [(board, move) for move in range(4)])
        bestmove, bestscore = max(enumerate(scores), key=lambda x: x[1])
        if bestscore == 0:
            return -1
        return bestmove
else:
    def find_best_move(m):
        board = to_c_board(m)
        return ailib.find_best_move(board)


KEY_TO_VAL = {
    0: 0,
    1: 2,
    2: 3,
    3: 1
}


def generate_replay_memory(replay_memory, env, cut_off, number_of_episode):

    for no_episode in range(number_of_episode):
        state = env.reset()

        for t in count(1):

            log2_state = np.clip(np.log2(state), 0.0, 18.0).tolist()
            move = find_best_move(log2_state)
            action = KEY_TO_VAL[move]

            next_state, reward, done, info = env.step(action)

            assert np.max(state) >= 2 and np.max(next_state) >= 2
            replay_memory.push(state, action, next_state, reward)
            max_tile = np.max(next_state)

            if max_tile >= cut_off or done or (len(replay_memory) >= replay_memory.max_length):
                logger.info('---'*8)
                logger.info('--- Episode: {}, Length of RepMemory: {}, Steps: {}, Cutoff: {}'.format(
                    no_episode, len(replay_memory), t, cut_off))
                break

            state = next_state

        if no_episode % (number_of_episode // 10) == 0:
            pickle.dump(replay_memory, open('replay_memory.p', 'wb'))
