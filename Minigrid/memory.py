import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
