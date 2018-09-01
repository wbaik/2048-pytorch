import torch
import torch.nn as nn
import torch.nn.functional as F


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


class XceptionLikeDuelingDQN(nn.Module):
    def __init__(self, n_action_space):
        super(XceptionLikeDuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            XceptionLike(64),
            Flatten(),
            nn.Linear(2880, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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


class XceptionLike(nn.Module):
    def __init__(self, in_channels):
        super(XceptionLike, self).__init__()

        self.branch2x2 = BasicConv2d(in_channels, 128, kernel_size=2, groups=in_channels)
        self.branch3x3 = BasicConv2d(in_channels, 128, kernel_size=3, groups=in_channels, padding=1)
        self.maxpool3x3 = nn.MaxPool2d(2, 1)
        self.branch4x4 = BasicConv2d(in_channels, 64, kernel_size=4, groups=in_channels, padding=1)

    def forward(self, x):
        branch2x2 = self.branch2x2(x)
        branch3x3 = self.branch3x3(x)
        branch3x3 = self.maxpool3x3(branch3x3)
        branch4x4 = self.branch4x4(x)

        outputs = [branch2x2, branch3x3, branch4x4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
