import time
import numpy as np
import torch
from itertools import count

import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper

from dqn import DQN_Agent, Memory, device
from ppo import PPOAgent
from utils import save_data, plot_average

import argparse

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

env_list = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-DoorKey-6x6-v0']
env_short_name_list = ['mpt6x6', 'rnd6x6', 'dk6x6']
env_index = 2
env_name = env_list[env_index]
env_short_name = env_short_name_list[env_index]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#DQN agent

mem_size = 100000
seed = 12
epsilon = 1.0
epsilon_min = 0.05
update_epsilon = 5
epsilon_decay = 0.97
batch_size= 2056
target_update = 10
episodes = 1000

gamma = 0.99
learning_rate = 0.0001
hidden_shape = 256
tau = 0.005

dqn_agent = DQN_Agent(env_name, seed, mem_size, gamma, epsilon, epsilon_min, update_epsilon, epsilon_decay, batch_size, target_update, hidden_shape, learning_rate, tau)
'''
episodes = 1000

for i in range(episodes):
    state = dqn_agent.env.reset()
    steps = 0
    dqn_agent.current_episode.append(i + 1)
    for t in count():
        steps += 1
        dqn_agent.env.render()
        action = dqn_agent.select_action(state, episodes)
        next_state, reward, done, _ = dqn_agent.env.step(action)

        if done and steps != dqn_agent.env.max_steps:
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
    avg_score = np.mean(dqn_agent.episode_scores[-100:])
        
    print('Episode: ', i + 1, ' Score: %.1f' % reward, ' Avg Score: %.1f' % avg_score, ' Steps done: ', dqn_agent.steps_done, ' Exploration: %.1f' % dqn_agent.epsilon)

    if i % target_update == 0:
        dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
    if dqn_agent.current_episode[-1] % dqn_agent.update_epsilon == 0:                        # steps_done changed to episodes done
            dqn_agent.epsilon *= dqn_agent.epsilon_decay
            dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon) 

dqn_agent.env.close()
print("Training Complete!")
dqn_agent.save_data(env_short_name, 5)
'''
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PPO agent

env = FlatObsWrapper(gym.make(env_name))
env.seed(seed)
N = 4096
batch_size = 128
n_epochs = 10
alpha = 0.0003

ppo_agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)

episodes = 1000
best_score = env.reward_range[0]
score_history = []
current_episode = []
learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(episodes):
    obs = env.reset()
    done = False
    score = 0
    current_episode.append(i+1)
    while not done:
        env.render()
        action, prob, val = ppo_agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        ppo_agent.remember(obs, action, prob, val, reward, done)
        if n_steps % N == 0:
            ppo_agent.learn()
            learn_iters += 1
        obs = obs_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score

    print('Episode: ', i + 1, ' Score: %.1f' % score, ' Avg Score: %.1f' % avg_score, ' Steps done: ', n_steps, 'Learning Steps done: ', learn_iters)

save_data(episodes=current_episode, scores=score_history, alg='ppo', short_name=env_short_name, run=0)