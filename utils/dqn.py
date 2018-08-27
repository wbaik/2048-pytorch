import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DQN(nn.Module):

    def __init__(self, n_action_space):
        super(DQN, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(1, 64, (1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
        )
        self.act = nn.Linear(32, n_action_space)

    def forward(self, x):
        x = self.s1(x)
        return self.act(x)

    def predict(self, x):
        '''
        :param x: input
        :return: max_value, index of Q[s,a]

        example
        -------
        >>> one_by_three = torch.randint(0,10,(1,3))
        tensor([[4, 2, 3]])
        >>> value, index = one_by_three.max(1)
        tensor[4.], tensor[0]
        '''
        return self.forward(x).max(1)
