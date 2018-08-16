from tkinter import *

from utils.logic import *
from utils import device, train_policy_with_a_batch, get_state

from itertools import count
import logging
import numpy as np
from random import choice

import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('progress.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"
KEY_CHOICES = (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT)


class GameGrid(Frame):
    def __init__(self, replay_memory, policy, target, optimizer,
                 epsilon=1.0, min_epsilon=0.2, eps_decay_rate=1e-4,
                 update_every=40, n_train=4000,
                 batch_size=1024, gamma=0.999):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                         KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right}

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        # DQN Configs
        self.replay_memory = replay_memory
        self.policy = policy
        self.target = target
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_decay_rate = eps_decay_rate
        self.update_every = update_every
        self.n_train = n_train
        self.batch_size = batch_size
        self.gamma = gamma

        self._dqn_play()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):

        if hasattr(self, 'matrix'):
            del self.matrix

        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number),
                                                    bg=BACKGROUND_COLOR_DICT[new_number],
                                                    fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def key_down(self, key):

        score = -1

        if key in self.commands:
            self.matrix, done, score = self.commands[key](self.matrix)

            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()

                if game_state(self.matrix)=='win':
                    self.grid_cells[1][1].configure(text="You",
                                                    bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!",
                                                    bg=BACKGROUND_COLOR_CELL_EMPTY)
                if game_state(self.matrix)=='lose':
                    self.grid_cells[1][1].configure(text="You",
                                                    bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!",
                                                    bg=BACKGROUND_COLOR_CELL_EMPTY)
                    return True, score

        return False, score

    def _dqn_play(self):

        def epsilon_greedy_action(state, epsilon, action_space=4):
            random_prob = np.random.rand()
            return choice(range(action_space)) if random_prob < epsilon \
                else self.policy.predict(state)[1].item()

        def adjust_epsilon(epsilon):
            epsilon *= (1 - self.eps_decay_rate)
            return max(epsilon, self.min_epsilon)

        def adjust_reward(r):
            reward = np.log2(r) / 10.0
            reward = reward[np.newaxis, ...]
            return torch.from_numpy(reward).float().clamp(0.0).to(device)

        max_all = 0.0
        max_reward_avg = 0.0

        epsilon = self.epsilon

        for i_episode in range(self.n_train + 1):

            for t in count(1):
                epsilon = adjust_epsilon(epsilon)

                # Compute State, Action, Reward ...
                state = get_state(self.matrix)
                action = epsilon_greedy_action(state, epsilon)
                done, reward_ = self.key_down(KEY_CHOICES[action])

                reward = adjust_reward(reward_)
                action = torch.tensor([action], device=device, dtype=torch.long)

                # Compute Next_State
                next_state = get_state(self.matrix) if not done else None

                # If there is no change in the states, penalize
                if next_state is not None and (state == next_state).all():
                    reward[...] = -3.0

                self.replay_memory.push(state, action, next_state, reward)

                # batch_size should be bound by the size of the replay_memory
                train_policy_with_a_batch(self.replay_memory, self.policy, self.target,
                                          self.batch_size, self.optimizer, self.gamma)

                if done:
                    max_reward = np.max(self.matrix)
                    max_reward_avg += max_reward
                    max_all = max(max_reward, max_all)

                    logger.debug('Max Tile in this iteration of {}: {}, Eps: {:.5f}'.format(
                                    t, max_reward, epsilon))
                    break

            self.init_matrix()

            if i_episode % self.update_every == 0:
                self.target.load_state_dict(self.policy.state_dict())
                logger.info('----------------------------------------------------')
                logger.info('Ending {} episode, epsilon  : {:.5f}'.format(i_episode, epsilon))
                logger.info('Max Tile Avg in {}th update : {} '.format(int(i_episode/self.update_every),
                                                                           max_reward_avg / self.update_every))
                logger.info('Max Tile Found              : {} '.format(max_all))

                max_reward_avg = 0.0
                max_all = 0.0

        self.quit()
