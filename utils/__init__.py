from utils.dqn import DQN, DuelingDQN, XceptionLikeDuelingDQN, SupervisedModel
from utils.environment import device
from utils.mc_search import direction
from utils.merge_game import MERGE_FUNCTIONS
from utils.replay_memory import ReplayMemory, Transition
from utils.state import get_state
from utils.train import train_dqn, train_supervised

