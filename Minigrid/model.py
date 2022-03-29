import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_shape):
        super(DQN, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(state_size, hidden_shape),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_shape, hidden_shape)),
            nn.ReLU(),
            nn.Linear(hidden_shape, action_size),
            nn.Identity()
        )

    def forward(self, x):
        q = self.Q(x)
        return q

