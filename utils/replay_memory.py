from collections import deque, namedtuple
import random

# class Transition:
#     def __init__(self, *args):
#         assert len(args) == 4
#         self.state, self.action, self.next_state, self.reward = args
#
#     def __repr__(self):
#         return 'state: {}, action: {}, next_state: {}, reward : {}'.format(
#             self.state, self.action, self.next_state, self.reward)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, max_length):
        self.max_length = max_length
        self.memory = deque()

    def push(self, *args):
        while len(self.memory) >= self.max_length:
            self.memory.popleft()
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __repr__(self):
        if len(self) == 0:
            return 'ReplayMemory.memory: EMPTY'
        return 'ReplayMemory.memory: {}...'.format(self.memory[0])

    def __len__(self):
        return len(self.memory)

