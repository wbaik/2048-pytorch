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
        self.act = nn.NoisyLinear(32, n_action_space)

    def forward(self, x):
        x = self.s1(x)
        return self.act(x)

    def predict(self, x):
        return self.forward(x).detach().sort(dim=1, descending=True)


class Combine(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Combine, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size) #, groups=in_channels)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concat, self).__init__()
        self.c1 = Combine(in_channels, out_channels, (2, 1))
        self.c2 = Combine(in_channels, out_channels, (1, 2))

        self.flat = Flatten()

    def forward(self, x):
        x1 = self.flat(self.c1(x))
        x2 = self.flat(self.c2(x))

        return torch.cat([x1, x2], 1)


class SupervisedModel(nn.Module):

    def __init__(self, n_action_space):
        super(SupervisedModel, self).__init__()
        self.c1 = Combine(1, 256, (2, 1))
        self.c2 = Combine(1, 256, (1, 2))

        self.concat1 = Concat(256, 1024)
        self.concat2 = Concat(256, 1024)

        self.l1 = nn.Linear(34816, 4096)
        self.l2 = nn.Linear(4096, 1024)
        self.l3 = nn.Linear(1024, 32)
        self.act = nn.Linear(32, n_action_space)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)

        x1 = self.concat1(x1)
        x2 = self.concat2(x2)

        x = torch.cat([x1, x2], 1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        return self.act(x)

    def predict(self, x):
        return self.forward(x).detach().sort(dim=1, descending=True)

    def get_argmax(self, x):
        return self.forward(x)[0].detach().argmax().item()


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
            nn.Conv2d(1, 1024, (1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            XceptionLike(1024),
            Flatten(),
            nn.Linear(27648, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(1024, 128),
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

        self.branch2x2 = BasicConv2d(in_channels, 1024, kernel_size=2, groups=in_channels)
        self.branch3x3 = BasicConv2d(in_channels, 1024, kernel_size=3, groups=in_channels, padding=1)
        self.maxpool3x3 = nn.MaxPool2d(2, 1)
        self.branch4x4 = BasicConv2d(in_channels, 1024, kernel_size=4, groups=in_channels, padding=1)

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


# https://github.com/Kaixhin/Rainbow/blob/master/model.py#L9-L46
# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        #         Notice self.weight_epsilon in `forward`
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
