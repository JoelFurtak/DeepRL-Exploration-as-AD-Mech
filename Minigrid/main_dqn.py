import time
import numpy as np
import torch
from itertools import count
import math

import gym
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper

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

env_list = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-DoorKey-6x6-v0', 'MiniGrid-MultiRoom-N4-S5-v0', 'MiniGrid-KeyCorridorS3R2-v0']
env_short_name_list = ['mpt6x6', 'rnd6x6', 'dk6x6', 'multi', 'keyco']
env_index = 4
env_name = env_list[env_index]
env_short_name = env_short_name_list[env_index]
seed = 123

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#DQN agent

env = gym.make(env_name)
env = FlatObsWrapper(env)
env.seed(seed)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(f'states_size: {state_size}')

mem_size = 100000
epsilon = 0.90
epsilon_min = 0.05
update_epsilon = 5
epsilon_decay = 0.97
batch_size= 2056
target_update = 10

gamma = 0.99
learning_rate = 0.0001
hidden_shape = 256
tau = 0.005

dqn_agent = DQN_Agent(state_size, action_size, mem_size, gamma, epsilon, batch_size, target_update, hidden_shape, learning_rate, tau)

episodes = 500
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
total_rewards = np.zeros(episodes)
total_int_reward = []
ep_total_int_reward = []

#if rnd:
#    agent = rnd_agent
#    total_int_reward = []
#    ep_total_int_reward = []
dqn_agent.load(alg='dqn', env_name=env_short_name, run=1)

for i in range(episodes):
    state = env.reset()
    steps = 0
    current_episode.append(i + 1)
    collisions = 0
    pick_up = 0
    drop = 0
    toggle = 0
    key_pickup = 0
    key_drop = 0
    door_toggle = 0
    turns = 0
    int_reward = 0
    for t in count():
        steps += 1
        #env.render()
        action = dqn_agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        if done and steps != env.max_steps:
            done_win = True
        else:
            done_win = False
        
        dqn_agent.remember(state, action, next_state, reward, done_win)

        if ((action == 0) or (action == 1)):
            turns += 1
        if ((action == 2) and (np.array_equal(state, next_state))):
            collisions += 1
        if (action == 3):
            pick_up += 1
        if ((action == 3) and (not np.array_equal(state, next_state))):
            key_pickup += 1
        if (action == 4):
            drop += 1
        if ((action == 4) and (not np.array_equal(state, next_state))):
            key_drop += 1
        if (action == 5):
            toggle += 1
        if ((action == 5) and (not np.array_equal(state, next_state))):
            door_toggle += 1
    
        n_steps += 1
        state = next_state
        dqn_agent.learn()
        if done:
            score_history.append(reward)
            collision_counter.append(collisions)
            pick_up_counter.append(pick_up)
            drops_counter.append(drop)
            toggles_counter.append(toggle)
            turn_counter.append(turns)
            key_pickups.append(key_pickup)
            key_drops.append(key_drop)
            doors_toggled.append(door_toggle)
            ep_total_int_reward.append(int_reward)
            break
    avg_score = np.mean(score_history[-100:])
        
    #print('Episode: ', i + 1, ' Score: %.1f' % reward, ' Avg Score: %.1f' % avg_score, ' Steps done: ', dqn_agent.steps_done, ' Exploration: %.1f' % dqn_agent.epsilon)
    print(f'Episode: {i+1}, Score: {reward:.2f}, Avg. Score: {avg_score:.2f}, Steps Done: {n_steps}, Epsilon: {dqn_agent.epsilon:.2f}')

    if i % target_update == 0:
        dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
    if current_episode[-1] % update_epsilon == 0:                        # steps_done changed to episodes done
            dqn_agent.epsilon *= epsilon_decay
            dqn_agent.epsilon = max(epsilon_min, dqn_agent.epsilon) 

env.close()
print("Training Complete!")
dqn_agent.save(alg='dqn', env_name=env_short_name, run=2)
save_data(episodes=current_episode, scores=score_history, collisions=collision_counter, pick_ups=pick_up_counter, drops=drops_counter, toggles=toggles_counter, key_pickups=key_pickups, key_drops=key_drops, door_toggles=doors_toggled, turns=turn_counter, intrinsic_reward=ep_total_int_reward,\
    alg='dqn', short_name=env_short_name, run=1)
