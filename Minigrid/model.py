from turtle import forward, hideturtle, st
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils import spectral_norm
from torch.nn import init
from torch.distributions.categorical import Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_shape):
        super(DQN, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(state_size, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, action_size),
            nn.Identity()
        )

    def forward(self, x):
        q = self.Q(x)
        return q

class SpectralDQN(nn.Module):
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


class PPOActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256):
        super(PPOActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    
class PPOCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(PPOCriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

class RNDPredictor(nn.Module):
    def __init__(self, input_dims, output_dims, learning_rate, hidden_dims=512):
        super(RNDPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(*input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, state):
        prediction = self.predictor(state)

        return prediction

class RNDTarget(nn.Module):
    def __init__(self, input_dims, output_dims, learning_rate, hidden_dims=512):
        super(RNDTarget, self).__init__()

        self.target = nn.Sequential(
            nn.Linear(*input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

        # untraining Network
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state):
        target = self.target(state)

        return target