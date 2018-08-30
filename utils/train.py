import numpy as np

import torch
import torch.nn.functional as F

from utils.replay_memory import Transition
from utils.environment import device


def train_dqn(policy_q, target_q, replay_memory, batch_size,
              optimizer, gamma, double_dqn):

    def load_game_from_replay_memory():
        transitions = replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def batch_to_tensor(given_batch, need_log2=False, action_batch=False):
        if need_log2:
            assert np.max(given_batch) >= 2, 'batch_data is weird: {}'.format(given_batch)
            given_batch = np.clip(np.log2(given_batch) / 10, 0, 18).tolist()

        dtype = torch.long if action_batch else torch.float32
        batch = list(map(lambda x: torch.tensor(x, device=device, dtype=dtype
                                                ).unsqueeze(0).unsqueeze(0), given_batch))
        return torch.cat(batch, 0)

    batch = load_game_from_replay_memory()

    state_batch = batch_to_tensor(batch.state, True)
    action_batch = batch_to_tensor(batch.action, action_batch=True)
    reward_batch = batch_to_tensor(batch.reward)
    next_state_batch = batch_to_tensor(batch.next_state, True)

    state_action_values = policy_q(state_batch).gather(1, action_batch)

    if double_dqn:
        next_state_action = policy_q(next_state_batch).detach().max(1)[1].unsqueeze(1)
        next_state_values_action_unspecified = target_q(next_state_batch).detach()
        next_state_values = next_state_values_action_unspecified.gather(1, next_state_action)

    else:
        next_state_values = target_q(next_state_batch).max(1)[0].detach().unsqueeze(1)

    expected_state_action_values = next_state_values * gamma + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_q.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
