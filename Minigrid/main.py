import time
import numpy as np
import torch
from itertools import count
import math

import gym
from gym import spaces
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper, FullyObsWrapper

from dqn import DQN_Agent, Memory, device
from ppo import PPOAgent
from rnd import RNDAgent
from utils import save_data, plot_average, Minigrid2Image

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

env_list = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-DoorKey-16x16-v0', 'MiniGrid-MultiRoom-N4-S5-v0', 'MiniGrid-KeyCorridorS3R2-v0', 'MiniGrid-LavaCrossingS9N2-v0']
env_short_name_list = ['mpt6x6', 'rnd6x6', 'dk16x16', 'multi', 'keyco', 'lava']
env_index = 5
env_name = env_list[env_index]
env_short_name = env_short_name_list[env_index]
seed = 1005

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PPO/ RND agent Training

env = gym.make(env_name)
env = FlatObsWrapper(env)
env.seed(seed)
N = 1024                                   # 4096 8192                       Multi 256 RND
batch_size = 256                           # 128 512 1024 2048               64
n_epochs = 4
alpha = 0.0003
alg = 'ppo'
train = False
play = True

#ppo_agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
#rnd_agent = RNDAgent(n_actions=env.action_space.n, batch_size=batch_size, lr=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)

novelid = False
rnd = False
if rnd:
    pass
#    agent = rnd_agent
else:
    pass
#    agent = ppo_agent

episodes = 5000
intrinsic_reward_coef = 0.00005
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
total_int_reward = []
ep_total_int_reward = []
total_rewards = np.zeros(episodes)

#curriculum = False
#if curriculum:
#   agent.load(alg=alg, env_name=env_short_name, run=1)

if train:
    for j in range(1, 6):
        seed = seed + j
        env.seed(seed)
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
        total_int_reward = []
        ep_total_int_reward = []
        death_counter = []
        total_steps = []
        total_rewards = np.zeros(episodes)
        agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)       
        for i in range(episodes):
            obs = env.reset()
            done = False
            ep_steps = 0
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
            ep_intrinsic_reward = 0
            intrinsic_reward = 0
            states = []
            death = False
            while not done:
                #env.render()
                states.append(obs)
                action, prob, val = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                if rnd:
                    intrinsic_reward = agent.intrinsic_reward(obs)
                    if (np.std(total_rewards) > 0):
                        intrinsic_reward /= np.std(total_rewards)
                    else:
                        intrinsic_reward = intrinsic_reward
                    total_int_reward.append(intrinsic_reward)
                    total_rewards = np.array(total_int_reward)
                    intrinsic_reward *= intrinsic_reward_coef                                            #0.00003
                    if novelid:
                        for state in states:
                            if np.array_equal(obs_, state):
                                intrinsic_reward *= 0
                    reward += intrinsic_reward
                n_steps += 1
                ep_steps += 1
                score += reward - intrinsic_reward
                ep_intrinsic_reward += intrinsic_reward
                agent.remember(obs, action, prob, val, reward, done)
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
                if(done and reward == 0 and ep_steps < env.max_steps):
                    death = True

                if n_steps % N == 0:
                    agent.learn()
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
            ep_total_int_reward.append(ep_intrinsic_reward)
            death_counter.append(death)
            total_steps.append(n_steps)

            if avg_score > best_score:
                best_score = avg_score

            print(f'Episode: {i+1}, Score: {score:.2f}, Intrinsic Reward: {ep_intrinsic_reward:.2f}, Avg Score: {avg_score:.2f}, Steps done: {n_steps}, Agent Died: {death}, Learning Steps done: {learn_iters}, Run: {j}') #, \nCollisions: {collisions}, Pick ups: {pick_up}, Drops: {drop}, Toggles: {toggle}, Keys picked up: {key_pickup}, Keys dropped: {key_drop}, Doors toggled: {door_toggle}, Turns: {turns}')

        agent.save(alg=alg, env_name=env_short_name, run=j)
        save_data(episodes=current_episode, steps=total_steps, scores=score_history, collisions=collision_counter, pick_ups=pick_up_counter, drops=drops_counter, \
            toggles=toggles_counter, key_pickups=key_pickups, key_drops=key_drops, door_toggles=doors_toggled, turns=turn_counter, intrinsic_reward=ep_total_int_reward, death=death_counter,\
            alg=alg, short_name=env_short_name, run=j, train=train)

if play:
    for j in range(1, 6):
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
        total_int_reward = []
        ep_total_int_reward = []
        death_counter = []
        total_steps = []
        total_rewards = np.zeros(episodes)
        agent = PPOAgent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
        agent.load(alg=alg, env_name='lava', run=j)
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
            ep_intrinsic_reward = 0
            intrinsic_reward = 0
            states = []
            death = False
            ep_steps = 0
            while not done:
                #env.render()
                states.append(obs)
                action, prob, val = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                if rnd:
                    intrinsic_reward = agent.intrinsic_reward(obs)
                    if (np.std(total_rewards) > 0):
                        intrinsic_reward /= np.std(total_rewards)
                    else:
                        intrinsic_reward = intrinsic_reward
                    total_int_reward.append(intrinsic_reward)
                    total_rewards = np.array(total_int_reward)
                    intrinsic_reward *= intrinsic_reward_coef                                            #0.00003
                    if novelid:
                        for state in states:
                            if np.array_equal(obs_, state):
                                intrinsic_reward *= 0
                    reward += intrinsic_reward
                    #print(f'debug: int reward: {intrinsic_reward}')
                n_steps += 1
                ep_steps += 1
                score += reward - intrinsic_reward
                ep_intrinsic_reward += intrinsic_reward
                #agent.remember(obs, action, prob, val, reward, done)
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
                if(done and reward == 0 and ep_steps < env.max_steps):
                    death = True
                
                #if n_steps % N == 0:
                #    agent.learn()
                #    learn_iters += 1
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
            ep_total_int_reward.append(ep_intrinsic_reward)
            death_counter.append(death)
            total_steps.append(n_steps)

            if avg_score > best_score:
                best_score = avg_score

            print(f'Episode: {i+1}, Score: {score:.2f}, Intrinsic Reward: {ep_intrinsic_reward:.2f}, Avg Score: {avg_score:.2f}, Steps done: {n_steps}, Agent Died: {death}, Run: {j}')

        save_data(episodes=current_episode, steps=total_steps, scores=score_history, collisions=collision_counter, pick_ups=pick_up_counter, drops=drops_counter, \
            toggles=toggles_counter, key_pickups=key_pickups, key_drops=key_drops, door_toggles=doors_toggled, turns=turn_counter, intrinsic_reward=ep_total_int_reward, death=death_counter,\
            alg=alg, short_name=env_short_name, run=j, train=train)