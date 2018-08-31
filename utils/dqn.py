import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DQN(nn.Module):

    def __init__(self, n_action_space):
        super(DQN, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (2, 2)),
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
        return self.forward(x).detach().sort(dim=1, descending=True)


class DuelingDQN(nn.Module):

    def __init__(self, n_action_space):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_action_space)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()

    def predict(self, x):
        return self.forward(x).detach().sort(dim=1, descending=True)
