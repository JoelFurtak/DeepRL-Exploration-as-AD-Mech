import random
import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

sns.set_theme(style="darkgrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda available: {}".format(torch.cuda.is_available()))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, next_state, reward)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Lin input connections depends on output of conv2d layers and therefore the input image size => compute it 
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh *32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one elem to determine next action, or a batch during optimization. Returns tensor ([[left0exp, right0exp]...])
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), - 1))

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')                              # Max Score: v0 = 200, v1 = 500
        self.state_size = self.env.observation_space.shape[0]           # Possible states
        self.action_size = self.env.action_space.n                      # Possible actions
        self.EPISODES = 500
        self.memory = ReplayMemory(2000)
        self.gamma = 0.95                                               # Reward discount
        self.epsilon = 1.0                                              # Exploration
        self.epsilon_min = 0.01                                         # Leftover Exploration when trained
        self.epsilon_decay = 0.9995                                      # Decay of epsilon atfer Episode
        #self.epsilon_decay = 250                                       # Decay of epsilon atfer Episode
        self.batch_size = 128                                           # Amount read out of memory
        self.target_update = 10                                         # Update Every 10 Steps
        self.steps_done = 0
        self.train_start = 1000

        self.resize = T.Compose([T.ToPILImage(), T.Resize(40, T.InterpolationMode.BICUBIC), T.ToTensor()])

        self.env.reset()
        self.init_screen = self.get_screen()
        _, _, self.screen_height, self.screen_width = self.init_screen.shape

        #self.model = DQN(self.screen_height, self.screen_width, self.action_size).to(device)
        #self.optimizer = optim.RMSprop(self.model.parameters())


        self.policy_net = DQN(self.screen_height, self.screen_width, self.action_size).to(device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.episode_scores = []

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)      # Middle of the Cart

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1)) # HWC --> CHW
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4): int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            sliced = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            sliced = slice(-view_width, None)
        else:
            sliced = slice(cart_location - view_width // 2, cart_location + view_width // 2)
        
        screen = screen[:, :, sliced]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255                   # RGB Max = 255
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0)

    def select_action(self, state):
        sample = random.random()
        action = None
        #epsilon_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        #print("Epsilon Thr.: {}".format(epsilon_threshold))
        if sample > self.epsilon:                                                                                   # Exploit
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)                                                # action = self.model(state).max(1)[1].view(1, 1)
        else:                                                                                                       # Explore
            action = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

        # Update Epsilon/ Exploration rate
        #self.epsilon *= self.epsilon_decay
        #self.epsilon = max(self.epsilon_min, self.epsilon)

        return action
        #if np.random.random() <= self.epsilon:
        #    return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
        #else:
        #    with torch.no_grad():
        #        return self.policy_net(state).max(1)[1].view(1, 1)

        
    #def seaborn(self):
        #x = self.episode_scores
        #y = torch.tensor(self.episode_scores, dtype=torch.float)
    #    sns.lineplot(data=torch.tensor(self.episode_scores, dtype=float))
    #    plt.show()

    def plot_scores(self):
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
        if len(self.memory) < self.batch_size:
            return                                                  # Not enough memories to sample from
        transitions = self.memory.sample(self.batch_size)
        #transitions = random.sample(self.memory, min(len(self.memory), self.batch_size))
        batch = tuple(zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[2])), device=device, dtype=torch.bool) #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch[2] if s is not None])
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[3])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)                      #self.policy_net(state_batch).gather(1, action_batch) self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()   #self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_value = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_value.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():                                                     #self.model.parameters(): 
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        #print(self.memory)
        #if len(self.memory) > self.train_start:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #self.epsilon = max(self.epsilon, self.epsilon_min)
        print("Epsilon: {}, Steps done: {}".format(self.epsilon, self.steps_done))

    def train(self):
        for i in range(self.EPISODES):
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            for t in count():
                self.env.render()
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                last_screen = current_screen 
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                
                self.remember(state, action, next_state, reward)
                #self.memory.push(state, action, next_state, reward)
                #print(self.epsilon)
                state = next_state
                self.optimize_model()
                if done:
                    self.episode_scores.append(t + 1)
                    break
            self.plot_scores()
            if i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.env.close()
        plt.ioff()
        plt.show()
        print("Training complete")

Agent = DQNAgent()
Agent.train()












