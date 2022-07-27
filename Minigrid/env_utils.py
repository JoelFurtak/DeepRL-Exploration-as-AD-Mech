import gym
import torch
from collections import deque, defaultdict
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

def _format_observation(obs):
    obs = torch.tensor(obs)
    return obs.view((1, 1) + obs.shape)