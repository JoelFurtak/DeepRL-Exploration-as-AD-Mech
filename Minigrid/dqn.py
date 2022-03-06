from turtle import forward
import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#env = FlatObsWrapper(gym.make('MiniGrid-Empty-5x5-v0'))
#env.reset()
#action_size = env.action_space.n
#print(action_size)

#for i in range(200):
#    action = np.random.randint(0, action_size)
#    env.step(action)
#    env.render()

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

class Memory(object):
    def __init__(self, state_size, max_size):
        self.max_size = max_size
        self.pointer = 0
        self.size = 0

        self.state = np.zeros((max_size, state_size))
        self.action = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_size))
        self.reward = np.zeros((max_size, 1))
        self.done_win = np.zeros((max_size, 1))
    
    def add(self, state, action, next_state, reward, done_win):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.next_state[self.pointer] = next_state
        self.reward[self.pointer] = reward
        self.done_win[self.pointer] = done_win

        self.pointer = (self.pointer + 1) % self.max_size
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

    def save(self):
        scaller = np.array([self.max_size, self.pointer, self.size], dtype=np.uint32)
        np.save("./buffer/scaller.npy", scaller)
        np.save("./buffer/state.npy", self.state)
        np.save("./buffer/action.npy", self.action)
        np.save("./buffer/next_state.npy", self.next_state)
        np.save("./buffer/reward.npy", self.reward)
        np.save("./buffer/done_win.npy", self.done_win)
    
    def load(self):
        scaller = np.load("./buffer/scaller.npy")

        self.max_size = scaller[0]
        self.pointer = scaller[1]
        self.size = scaller[2]

        self.state = np.load("./buffer/state.npy")
        self.action = np.load("./buffer/action.npy")
        self.next_state = np.load("./buffer/next_state.npy")
        self.reward = np.load("./buffer/reward.npy")
        self.done_win = np.load("./buffer/done_win.npy")


class DQN_Agent:
    def __init__(self):
        self.env = FlatObsWrapper(gym.make('MiniGrid-Empty-Random-6x6-v0'))
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.memory = Memory(self.state_size, max_size=20000)
        self.gamma = 0.99
        self.epsilon = 1.00
        self.epsilon_min = 0.05
        self.update_epsilon = 500
        self.epsilon_decay = 0.90
        self.batch_size = 1028
        self.target_update = 10
        self.steps_done = 0
        self.max_steps = 500

        self.hidden_shape = 250
        self.lr = 0.0001
        self.tau  = 0.005

        self.env.reset()

        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_shape).to(device)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_net = copy.deepcopy(self.policy_net)
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.current_episode = []
        self.episode_scores = []

    def select_action(self, state, episodes):
        print("Episode: {}/{}, Steps done: {}, epsilon %: {}".format(self.current_episode[-1], episodes, self.steps_done, self.epsilon))     
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.action_size)
            else:
                action = self.policy_net(state).argmax().item()
        if self.steps_done % self.update_epsilon == 0:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        return action

    def seaborn(self):
        episode = np.array(self.current_episode)
        scores = np.array(self.episode_scores)
        df = {'episode': episode, 'score': scores}
        pdscores = pd.DataFrame(df)
        #pdscores.to_csv('./data/3k_random_steps/run#3/results.csv')

        sns.lineplot(x='episode', y='score',data=pdscores)

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

    def train(self, episodes):
        for i in range(episodes):
            state = self.env.reset()
            steps = 0
            self.current_episode.append(i + 1)
            for t in count():
                steps += 1
                self.env.render()
                action = self.select_action(state, episodes)
                next_state, reward, done, _ = self.env.step(action)

                if done and steps != self.max_steps:
                    done_win = True
                else:
                    done_win = False

                self.memory.add(state, action, next_state, reward, done_win)
                self.steps_done += 1
                state = next_state
                self.optimize_model()
                if done:
                    self.episode_scores.append(reward)
                    break
            if i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.env.close()
        print("Training Complete!")
        self.seaborn()
        plt.ioff()
        plt.show()

    def save(self, algo, env_name, episodes):
        torch.save(self.policy_net.state_dict(), "./model/{}_{}_{}.pth".format(algo, env_name, episodes))

    def load(self, algo, env_name, episodes):
        self.policy_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo, env_name, episodes)))        
        self.target_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo, env_name, episodes)))



Agent = DQN_Agent()
Agent.train(500)

