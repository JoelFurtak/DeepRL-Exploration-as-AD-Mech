import time
import numpy as np
import torch
from itertools import count
import math

import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper

from dqn import DQN_Agent, Memory, device
from ppo import PPOAgent
from rnd import RNDAgent
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
seed = 123
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
#PPO/ RND agent

env = FlatObsWrapper(gym.make(env_name))
env.seed(seed)
N = 4096
batch_size = 128
n_epochs = 10
alpha = 0.0003

ppo_agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
rnd_agent = RNDAgent(n_actions=env.action_space.n, batch_size=batch_size, lr=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)

rnd = True
if rnd:
    agent = rnd_agent
    total_int_reward = []
    ep_total_int_reward = []
else:
    agent = ppo_agent

episodes = 500
best_score = env.reward_range[0]
score_history = []
current_episode = []
learn_iters = 0
avg_score = 0
n_steps = 0
collision_counter = []
pick_up_counter = []
drops_counter = []
toggles_counter = []
turn_counter = []
key_pickups = []
key_drops = []
doors_toggled = []
#reward_rs = RunningEstimateStd()
total_rewards = np.zeros(episodes)

for i in range(episodes):
    obs = env.reset()
    done = False
    score = 0
    current_episode.append(i+1)
    collisions = 0
    pick_up = 0
    drop = 0
    toggle = 0
    key_pickup = 0
    key_drop = 0
    door_toggle = 0
    turns = 0
    if rnd:
        ep_intrinsic_reward = 0
        intrinsic_reward = 0
    while not done:
        #env.render()
        action, prob, val = ppo_agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        #print(f'debug reward: {reward}')
        if rnd:
            intrinsic_reward = rnd_agent.intrinsic_reward(obs)
            #print(f'debug {intrinsic_reward}')
            #total_int_rewards = np.zeros(len(score_history))
            #mean, std, c = np.mean(total_rewards), np.std(total_rewards), len(score_history)
            #print(f'Debug mean... {mean}{std}{c}')
            #reward_rs.update(mean, std**2, c)
            if (np.std(total_rewards) > 0):
                intrinsic_reward /= np.std(total_rewards)
            else:
                intrinsic_reward = intrinsic_reward
            #print(f'debug {intrinsic_reward}')
            total_int_reward.append(intrinsic_reward)
            total_rewards = np.array(total_int_reward)
            intrinsic_reward *= 0.0001
            reward += intrinsic_reward
        #print(f'debug reward + int: {reward}')
        n_steps += 1
        score += reward
        ep_intrinsic_reward += intrinsic_reward
        #print(f'Debug: ep int reward : {ep_intrinsic_reward}')
        ppo_agent.remember(obs, action, prob, val, reward, done)
        if ((action == 0) or (action == 1)):
            turns += 1
        if ((action == 2) and (np.array_equal(obs, obs_))):
            collisions += 1
        if (action == 3):
            pick_up += 1
        if ((action == 3) and (not np.array_equal(obs, obs_))):
            key_pickup += 1
        if (action == 4):
            drop += 1
        if ((action == 4) and (not np.array_equal(obs, obs_))):
            key_drop += 1
        if (action == 5):
            toggle += 1
        if ((action == 5) and (not np.array_equal(obs, obs_))):
            door_toggle += 1
        
        if n_steps % N == 0:
            ppo_agent.learn()
            learn_iters += 1
        obs = obs_
    collision_counter.append(collisions)
    pick_up_counter.append(pick_up)
    drops_counter.append(drop)
    toggles_counter.append(toggle)
    turn_counter.append(turns)
    key_pickups.append(key_pickup)
    key_drops.append(key_drop)
    doors_toggled.append(door_toggle)
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if rnd:
        ep_total_int_reward.append(ep_intrinsic_reward)

    if avg_score > best_score:
        best_score = avg_score

    print(f'Episode: {i+1}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Steps done: {n_steps}, Learning Steps done: {learn_iters}') #, \nCollisions: {collisions}, Pick ups: {pick_up}, Drops: {drop}, Toggles: {toggle}, Keys picked up: {key_pickup}, Keys dropped: {key_drop}, Doors toggled: {door_toggle}, Turns: {turns}')

save_data(episodes=current_episode, scores=score_history, collisions=collision_counter, pick_ups=pick_up_counter, drops=drops_counter, toggles=toggles_counter, key_pickups=key_pickups, key_drops=key_drops, door_toggles=doors_toggled, turns=turn_counter, intrinsic_reward=ep_total_int_reward,\
    alg='rnd', short_name=env_short_name, run=4)