import numpy as np

import torch

from utils.environment import device


def get_state(matrix):
    log_based = np.log2(matrix) / 10.0 # 10 for 1024
    log_based = log_based[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(log_based).float().clamp(0.0).to(device)


