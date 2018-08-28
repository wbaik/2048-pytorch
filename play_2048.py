from itertools import count
import logging
import numpy as np
from random import choice
import torch
from utils import device, train_dqn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Play2048:
    def __init__(self, env, replay_memory, policy, target, optimizer,
                 batch_size=32, epsilon=1.0, eps_decay_rate=1e-5, min_epsilon=0.1,
                 n_train=1000000, update_every=10000, gamma=0.99, double_dqn=True):
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

    def play_2048(self, mode='train'):
        def epsilon_greedy_action(state, epsilon, action_space=4):
            random_prob = np.random.rand()
            state = np.clip(np.log2(state) / 10, 0, 15)[np.newaxis, np.newaxis, ...].tolist()
            state = torch.tensor(state, device=device)
            return choice(range(action_space)) if random_prob < epsilon \
                else self.policy.predict(state)[1].item()

        def adjust_epsilon(epsilon):
            epsilon *= (1 - self.eps_decay_rate)
            return max(epsilon, self.min_epsilon)

        def log_rewards(next_state, max_all, max_reward_avg, i_episode):
            max_reward = np.max(next_state)
            max_reward_avg += max_reward
            max_all = max(max_reward, max_all)
            logger.info('---------------------')
            logger.info('Episode no.:{}'.format(i_episode))
            logger.info('Game over, num_steps: {}, max_reward: {}'.format(t, max_reward))
            return max_all, max_reward_avg

        if mode != 'train':
            self.epsilon = self.min_epsilon = 0.01

        epsilon = self.epsilon

        max_all = 0.0
        max_reward_avg = 0.0

        for i_episode in range(1, self.n_train + 1):

            state = self.env.reset()

            for t in count(1):

                epsilon = adjust_epsilon(epsilon)
                action = epsilon_greedy_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                if t % 100 == 0:
                    self.replay_memory.push(state, action, next_state, reward)

                if mode == 'train':
                    train_dqn(self.policy, self.target, self.replay_memory,
                              self.batch_size, self.optimizer, self.gamma, self.double_dqn)

                if done:
                    max_all, max_reward_avg = log_rewards(next_state,
                                                          max_all,
                                                          max_reward_avg,
                                                          i_episode)
                    print('--- Game Over: Rendering...')
                    self.env.render('human')
                    break

            if i_episode % self.update_every == 0:
                self.target.load_state_dict(self.policy.state_dict())
                logger.info('---------------------')
                logger.info('Ending {} episodes, epsilon: {:.5f}'.format(i_episode, epsilon))
                logger.info('Max Tile Avg in {}th update: {}'.format(int(i_episode / self.update_every),
                                                                     max_reward_avg / self.update_every))
                logger.info('Max Tile Found             : {}'.format(max_all))
                max_reward_avg = 0.0
                max_all = 0.0