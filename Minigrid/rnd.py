import numpy as np 
import math
import gym 
import gym_minigrid
from memory import PPOMemory
from model import PPOActorNetwork, PPOCriticNetwork, RNDPredictor, RNDTarget

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.categorical import Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNDAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10, ent_coef=0.01, update_prop=0.25):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.update_prop = update_prop

        self.actor = PPOActorNetwork(n_actions, input_dims, lr)
        self.critic = PPOCriticNetwork(input_dims, lr)
        self.memory = PPOMemory(batch_size)

        self.predictor = RNDPredictor(input_dims, n_actions, 1e-4)
        self.target = RNDTarget(input_dims, n_actions, 1e-4)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def intrinsic_reward(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).to(device)

        target = self.target(observation)
        prediction = self.predictor(observation)

        intrinsic_reward = (target - prediction).pow(2).sum() / 2

        return intrinsic_reward.data.cpu().numpy()

    def learn(self):
        for i in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(device)            
            values = torch.tensor(values).to(device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(device)
                actions = torch.tensor(action_arr[batch]).to(device)

                # RND
                predict_next = self.predictor(states)
                target_next = self.target(states)

                forward_loss = nn.MSELoss(predict_next, target_next.detach()).mean(-1)
                mask = torch.rand(len(forward_loss)).to(device)
                mask = (mask < self.update_prop).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                # Entropy
                entropy = dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                self.predictor.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                self.predictor.optimizer.step()
        
        self.memory.clear_memory()
