from importlib.metadata import requires
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ComplicatedMemory:
    def __init__(self, max_size, state_size, batch_size):
        self.max_size = max_size
        self.state_size = state_size
        self.batch_size = batch_size

        self.pointer = 0
        self.size = 0
    
        self.states = np.zeros((self.max_size, state_size))
        self.actions = np.zeros((self.max_size, 1))
        self.probs = np.zeros((self.max_size, 1))
        self.rewards = np.zeros((self.max_size, 1))
        self.dones = np.zeros((self.max_size, 1))
        self.vals = np.zeros((self.max_size, 1))

    def add(self, state, action, probs, vals, reward, done):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.probs[self.pointer] = probs
        self.vals[self.pointer] = vals
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def generate_batches(self):
        n_states = self.states.shape[0]
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return self.states,\
                self.actions,\
                self.probs,\
                self.vals,\
                self.rewards,\
                self.dones,\
                batches
    
    def clear_memory(self):
        self.states = np.zeros((self.max_size, self.state_size))
        self.actions = np.zeros((self.max_size, 1))
        self.probs = np.zeros((self.max_size, 1))
        self.rewards = np.zeros((self.max_size, 1))
        self.dones = np.zeros((self.max_size, 1))
        self.vals = np.zeros((self.max_size, 1))