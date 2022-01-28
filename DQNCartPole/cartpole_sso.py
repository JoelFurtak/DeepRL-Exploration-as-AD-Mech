from distutils.log import info
import random
from turtle import forward
import gym
import copy
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pandas as pd

# Matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

sns.set_theme(style="darkgrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda available: {}".format(torch.cuda.is_available()))

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

    def forward(self, s):
        q = self.Q(s)
        return q

class ReplayBuffer(object):
    def __init__(self, state_size, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_size))
        self.action = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_size))
        self.reward = np.zeros((max_size, 1))
        self.done_win = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done_win):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done_win[self.ptr] = done_win

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        with torch.no_grad():
            return (
                torch.FloatTensor(self.state[index]).to(device),
                torch.Tensor(self.action[index]).long().to(device),
                torch.FloatTensor(self.next_state[index]).to(device),
                torch.FloatTensor(self.reward[index]).to(device),
                torch.FloatTensor(self.done_win[index]).to(device)
            )

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')                              # Max Score: v0 = 200, v1 = 500
        self.state_size = self.env.observation_space.shape[0]           # Possible states
        self.action_size = self.env.action_space.n                      # Possible actions
        self.EPISODES = 500
        self.memory = ReplayBuffer(self.state_size, max_size=10000)
        self.gamma = 0.99                                               # Reward discount
        self.explporation = 0.2                                                                                            
        self.update_exploration = 1000                                  # Update Exploration rate every x steps               
        self.explporation_decay = 0.99                                  # Exploration decay                                  
        self.batch_size = 128                                           # Amount read out of memory
        self.target_update = 10                                         # Update Every 10 Steps
        self.steps_done = 0
        self.random_steps = 3000                                        # Random Exploration steps to map Environment

        self.hidden_shape = 200                                         
        self.lr = 0.0001
        self.tau = 0.005

        self.env.reset()

        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_shape).to(device)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_net = copy.deepcopy(self.policy_net)
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.current_episode = []
        self.episode_scores = []

    def select_action(self, state):
        print("Episode: {}/{}, Steps done: {}, Exploration %: {}".format(self.current_episode[-1], self.EPISODES, self.steps_done, self.explporation))     
        if self.memory.size < self.random_steps:
            action = np.random.randint(0, self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                if np.random.rand() < self.explporation:
                    action = np.random.randint(0, self.action_size)
                else:
                    action = self.policy_net(state).argmax().item()
        if self.steps_done % self.update_exploration == 0:
            self.explporation *= self.explporation_decay
        return action

    def seaborn(self):
        '''Plot with Seaborn'''
        episode = np.array(self.current_episode)
        scores = np.array(self.episode_scores)
        avrg = np.array(np.mean(scores))
        d = {'Episode': episode, 'Score': scores, 'Average': avrg}
        pdscores = pd.DataFrame(d)
        #pdscores2 = pd.melt(pdscores, ['Episode'], value_name='Score')                     

        sns.lineplot(x='Episode', y='Score',data=pdscores)

        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())



    def plot_scores(self):
        '''Plot with matplotlib'''
        plt.figure(1)
        plt.clf()
        scores = torch.tensor(self.episode_scores, dtype=torch.float)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(scores.numpy())
        # 100 EP average
        if len(scores) >= 100:
            means = scores.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model(self):
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

    def train(self):
        for i in range(self.EPISODES):
            state = self.env.reset()
            steps = 0
            self.current_episode.append(i + 1)
            for t in count():
                steps += 1
                self.env.render()
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                if done and steps != self.env._max_episode_steps:
                    done_win = True
                else:
                    done_win = False
                
                self.memory.add(state, action, next_state, reward, done_win)
                self.steps_done += 1
                state = next_state
                self.optimize_model()
                if done:
                    self.episode_scores.append(t + 1)
                    break
            self.plot_scores()
            #self.seaborn()
            if i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.env.close()
        print("Training complete")
        plt.ioff()
        plt.show()

Agent = DQNAgent()
Agent.train()












