from copy import deepcopy
import datetime
from itertools import count
import gym
import gym_2048
from gym_2048.engine import Engine
import logging
import matplotlib.pyplot as plt
import numpy as np
from random import choice
import time
import torch
from utils import device, train_dqn, MERGE_FUNCTIONS
import seaborn as sns


FILE_NAME = 'training_{}.log'.format(datetime.datetime.now())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(FILE_NAME)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Play2048:
    def __init__(self, env, replay_memory, policy, target, optimizer,
                 batch_size=32, epsilon=1.0, eps_decay_rate=1e-5, min_epsilon=0.1,
                 n_train=1000000, update_every=10000, gamma=0.999, double_dqn=True):
        self.env = env
        self.replay_memory = replay_memory
        self.policy = policy
        self.target = target
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_decay_rate = eps_decay_rate
        self.min_epsilon = min_epsilon
        self.n_train = n_train
        self.update_every = update_every
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.max_tiles = []
        self.fig, self.axis = self.get_figure_and_axis()

    @classmethod
    def get_figure_and_axis(cls):
        plt.style.use(['ggplot', 'fivethirtyeight'])
        plt.ion()
        return plt.subplots(2, 1)

    @classmethod
    def log_rewards(cls, next_state, max_all, max_reward_avg, t, axis, max_tiles):
        max_reward = np.max(next_state)
        max_reward_avg += max_reward
        max_all = max(max_reward, max_all)

        logger.info('---------------------')
        logger.info('num_steps: {}, max_reward: {}'.format(t, max_reward))

        max_tiles += [max_reward.item()]
        axis[0].scatter(len(max_tiles), max_tiles[-1])
        plt.pause(0.01)

        return max_all, max_reward_avg, max_tiles, axis

    @classmethod
    def get_tensor_for_predictions(cls, state_original):
        torch_state = np.clip(np.log2(state_original) / 10, 0, 18)[np.newaxis, np.newaxis, ...].tolist()
        torch_state = torch.tensor(torch_state, device=device)
        return torch_state

    @classmethod
    def log_on_update(cls, _i_episode, _max_reward_avg, _axis, _max_tiles, update_every):
        sns.distplot(_max_tiles, ax=_axis[1])
        logger.info('Ending {} episodes'.format(_i_episode))
        logger.info('Max Tile Avg in {}th update: {}'.format(_i_episode // update_every,
                                                             _max_reward_avg / update_every))

    @classmethod
    def supervised_model_test(cls, model, n_train, update_every):


        env = gym.make('game-2048-v0')

        max_all, max_reward_avg = 0.0, 0.0
        max_tiles = []
        _, axis = cls.get_figure_and_axis()

        for i_episode in range(1, n_train + 1):

            state = env.reset()

            for t in count(1):

                _, action = cls.best_action(state, model)

                next_state, reward, done, info = env.step(action)

                if done:
                    env.render('human')
                    max_all, max_reward_avg, max_tiles, axis = cls.log_rewards(next_state, max_all,
                                                                               max_reward_avg, t,
                                                                               axis, max_tiles)
                    break

            if i_episode % update_every == 0:
                cls.log_on_update(i_episode, max_reward_avg, axis, max_tiles, update_every)
                max_all, max_reward_avg = 0.0, 0.0

    @classmethod
    def get_actions_available(cls, given_state):
        dummy_engine = Engine()
        dummy_engine.board = given_state
        return dummy_engine.moves_available()


    @classmethod
    def best_action(cls, given_state, model):
        torch_state = cls.get_tensor_for_predictions(given_state)
        actions_available = cls.get_actions_available(given_state)

        pred_value, pred_action = model.predict(torch_state)
        convert_dim_to_one = lambda x: x[0].tolist()
        pred_value, pred_action = list(map(convert_dim_to_one, [pred_value, pred_action]))

        for value, action in zip(pred_value, pred_action):
            if actions_available[action]:
                return value, action

    def play_2048(self, mode='train'):

        def epsilon_greedy_action(state, action_space=4):
            '''
            :param state: list of shape [4, 4]
            :param action_space: constant 4
            :return: value of an action in [0...3]
            '''

            def get_actions_available(given_state):
                dummy_engine = Engine()
                dummy_engine.board = given_state
                return dummy_engine.moves_available()

            def get_tensor_for_predictions(state_original):
                torch_state = np.clip(np.log2(state_original) / 10, 0, 18)[np.newaxis, np.newaxis, ...].tolist()
                torch_state = torch.tensor(torch_state, device=device)
                return torch_state

            def best_action(given_state):
                torch_state = get_tensor_for_predictions(given_state)
                actions_available = get_actions_available(given_state)

                pred_value, pred_action = self.policy.predict(torch_state)
                convert_dim_to_one = lambda x: x[0].tolist()
                pred_value, pred_action = list(map(convert_dim_to_one, [pred_value, pred_action]))

                for value, action in zip(pred_value, pred_action):
                    if actions_available[action]:
                        return value, action

            def one_step_look_ahead(original_state):
                dummy_engine = Engine()
                dummy_engine.board = original_state

                current_best_value, current_best_action = 0.0, -1
                actions_available = dummy_engine.moves_available()
                for idx in range(4):
                    if actions_available[idx]:
                        log2_reward, ended = dummy_engine.move(idx)
                        if not ended:
                            this_value, this_action = best_action(dummy_engine.board)
                            this_value += log2_reward

                            if current_best_value < this_value:
                                current_best_value, current_best_action = this_value, idx

                return current_best_value, current_best_action

            assert np.max(state) >= 2.0

            if mode == 'test' and False: # Ignoring for the time being
                state_copy = deepcopy(state)
                action_one_step_ahead = one_step_look_ahead(state_copy)[1]
                return action_one_step_ahead if action_one_step_ahead != -1 else best_action(state)[1]

            return choice(range(action_space)) if np.random.rand() < self.epsilon else best_action(state)[1]

        def adjust_epsilon():
            self.epsilon *= (1 - self.eps_decay_rate)
            self.epsilon = max(self.epsilon, self.min_epsilon)

        def log_rewards(next_state, max_all, max_reward_avg):
            max_reward = np.max(next_state)
            max_reward_avg += max_reward
            max_all = max(max_reward, max_all)

            if mode == 'train':
                logger.info('---------------------')
                logger.info('Episode no.:{}'.format(i_episode))
                logger.info('Game over, num_steps: {}, max_reward: {}, epsilon: {:.5f}'.format(t, max_reward, self.epsilon))

            self.max_tiles += [max_reward.item()]
            self.axis[0].scatter(len(self.max_tiles), self.max_tiles[-1])
            plt.pause(0.001)

            return max_all, max_reward_avg

        def log_on_update_weights():
            logger.info('---------------------')
            logger.info('Ending {} episodes, epsilon: {:.5f}'.format(i_episode, self.epsilon))
            logger.info('Max Tile Avg in {}th update: {}'.format(i_episode // self.update_every,
                                                                 max_reward_avg / self.update_every))
            logger.info('Max Tile Found             : {}'.format(max_all))
            sns.distplot(self.max_tiles, ax=self.axis[1])

        if mode != 'train':
            self.epsilon = self.min_epsilon = 1e-8

        max_all, max_reward_avg = 0.0, 0.0

        for i_episode in range(1, self.n_train + 1):

            state = self.env.reset()

            for t in count(1):

                adjust_epsilon()
                action = epsilon_greedy_action(state)
                next_state, reward, done, info = self.env.step(action)

                if t % 100 == 0:
                    self.replay_memory.push(state, action, next_state, reward)

                if mode == 'train':
                    train_dqn(self.policy, self.target, self.replay_memory,
                              self.batch_size, self.optimizer, self.gamma, self.double_dqn)
                # else:
                #     self.env.render('human')
                #     time.sleep(0.05)

                if done:
                    self.env.render('human')
                    max_all, max_reward_avg = log_rewards(next_state, max_all, max_reward_avg)
                    break

            if i_episode % self.update_every == 0:
                self.target.load_state_dict(self.policy.state_dict())
                log_on_update_weights()
                max_all, max_reward_avg = 0.0, 0.0
