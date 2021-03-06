from turtle import forward
import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import count

import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from memory import Memory
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN_Agent:
    def __init__(self, state_size, action_size, mem_size, gamma, epsilon, batch_size, target_update, hidden_shape, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = Memory(self.state_size, max_size=mem_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update

        self.steps_done = 0

        self.hidden_shape = hidden_shape
        self.lr = lr
        self.tau  = tau

        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_shape).to(device)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_net = copy.deepcopy(self.policy_net)
        for p in self.target_net.parameters():
            p.requires_grad = False

    def remember(self, state, action, next_state, reward, done_win):
        self.memory.add(state, action, next_state, reward, done_win)

    def choose_action(self, state):     
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.action_size)
            else:
                action = self.policy_net(state).argmax().item()
        return action

    def learn(self):
        state, action, next_state, reward, done_win = self.memory.sample(self.batch_size)
        
        with torch.no_grad():
            max_q_prime = self.target_net(next_state).max(1)[0].unsqueeze(1)
        
        target_Q = reward + (1 - done_win) * self.gamma * max_q_prime

        current_q = self.policy_net(state)
        current_q_a = current_q.gather(1, action)
        
        
        loss = F.mse_loss(current_q_a, target_Q)
        self.policy_net_optimizer.zero_grad()
        loss.backward()
        self.policy_net_optimizer.step()

        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):                                                     
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, alg, env_name, run):
        torch.save(self.policy_net.state_dict(), f"./model/{alg}_{env_name}_{run}.pth")

    def load(self, alg, env_name, run):
        self.policy_net.load_state_dict(torch.load(f"./model/{alg}_{env_name}_{run}.pth"))        
        self.target_net.load_state_dict(torch.load(f"./model/{alg}_{env_name}_{run}.pth"))
