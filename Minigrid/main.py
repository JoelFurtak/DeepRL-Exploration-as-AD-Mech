import time
import numpy
import torch
from itertools import count

import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper

from dqn import DQN_Agent, Memory, device

import argparse
from datetime import datetime

# Parameters
#parser = argparse.ArgumentParser()
#parser.add_argument('--seed', type=int, default=123, help='Random seed.')
#parser.add_argument('--Max_train_steps', type=int, default=1000, help='Max training steps.')
#parser.add_argument('--save_interval', type=int, default=50000, help='Model saving interval, in steps.')
#parser.add_argument('--eval_interval', type=int, default=1000, help='Model evaluating interval, in steps.')
#parser.add_argument('--random_steps', type=int, default=3000, help='Steps for radnom policy to explore.')
#parser.add_argument('--update_every', type=int, default=10, help='Training frequency.')

#parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor.')
#parser.add_argument('--net_width', type=int, default=250, help='Hidden net width.')
#parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
#parser.add_argument('--mem_size', type=int, default=20000, help='Memory size for Replay Buffer.')
#parser.add_argument('--epsilon', type= float, default=1.0, help='Exploration rate.')
#args = parser.parse_args()

env_list = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-FourRooms-v0']
env_short_name_list = ['mpt6x6', 'rnd6x6', '4rooms']
env_index = 1
env_name = env_list[env_index]
env_short_name = env_short_name_list[env_index]

mem_size = 20000
seed = 123
epsilon = 1.0
epsilon_min = 0.05
update_epsilon = 500
epsilon_decay = 0.95
batch_size= 1028
target_update = 10

gamma = 0.99
learning_rate = 0.0001
hidden_shape = 35
tau = 0.005

dqn_agent = DQN_Agent(env_name, seed, mem_size, gamma, epsilon, epsilon_min, update_epsilon, epsilon_decay, batch_size, target_update, hidden_shape, learning_rate, tau)
dqn_agent.train(500)

# Training Loop
'''
episodes = 500

for i in range(episodes):
    state = dqn_agent.env.reset()
    steps = 0
    dqn_agent.current_episode.append(i + 1)
    for t in count():
        steps += 1
        dqn_agent.env.render()
        action = dqn_agent.select_action(state, episodes)
        next_state, reward, done, _ = dqn_agent.env.step(action)

        if done and steps != dqn_agent.max_steps:
            done_win = True
        else:
            done_win = False
        
        dqn_agent.memory.add(state, action, next_state, reward, done_win)
        dqn_agent.steps_done += 1
        state = next_state
        dqn_agent.optimize_model()
        if done:
            dqn_agent.episode_scores.append(reward)
            break
    if i % target_update == 0:
        dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
    dqn_agent.env.close()
    print("Training Complete!")
    dqn_agent.seaborn()
'''