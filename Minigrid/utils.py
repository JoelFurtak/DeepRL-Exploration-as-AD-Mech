import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
import gym
#sns.set_style(style='darkgrid')

def save_data(episodes, steps, scores, collisions, pick_ups, drops, toggles, key_pickups, key_drops, door_toggles, turns, intrinsic_reward, death, alg, short_name, run, train):
    episodes = np.array(episodes)
    scores = np.array(scores)
    collisions = np.array(collisions)
    pick_ups = np.array(pick_ups)
    drops = np.array(drops)
    toggles = np.array(toggles)
    key_pickups = np.array(key_pickups)
    key_drops = np.array(key_drops)
    door_toggles = np.array(door_toggles)
    turns = np.array(turns)
    intrinsic_reward = np.array(intrinsic_reward)
    death = np.array(death)
    steps = np.array(steps)

    print('... saving data ...')
    df = {
        'episode': episodes,
        'steps': steps,
        'score': scores, 
        'collisions': collisions, 
        'pick_ups': pick_ups, 
        'drops': drops, 
        'toggles': toggles, 
        'key_pickups': key_pickups, 
        'key_drops': key_drops, 
        'door_toggles':door_toggles, 
        'turns':turns, 
        'intrinsic_reward':intrinsic_reward, 
        'death':death
        }
    df = pd.DataFrame(df)
    if train:
        df.to_csv(f'./data/{alg}/{short_name}/run#{run}/results_train.csv')
    else:
        df.to_csv(f'./data/{alg}/{short_name}/run#{run}/results_play.csv')

def plot_scores(alg, short_name, run):
    df = pd.read_csv(f'./data/{alg}/{short_name}/run#{run}/results.csv')

    sns.lineplot(data=df, x='episode', y='score')
    plt.show()

def plot_average(alg, short_name, run):
    df = pd.read_csv(f'./data/{alg}/{short_name}/run#{run}/results.csv')
    df.score = df.score.rolling(100, min_periods=1).mean()

    sns.lineplot(data=df, x='episode', y='score')
    plt.show()

def plot_mlt_average(alg, short_names):
    dfs = []
    for name in short_names:
        for i in range(6):
            try:                                                        # i = 1..5
                df = pd.read_csv(f'./data/{alg}/{name}/run#{i}/results.csv', index_col=False)
                df.score = df.score.rolling(100).mean()
                df['Parameters'] = name
                dfs.append(df)
            except Exception as e:                                      # i = 0
                pass
    
    df = pd.concat(dfs).reset_index()

    sns.lineplot(data=df, x='episode', y='score', hue='Parameters' , palette='husl', ci='sd')
    plt.show()

class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, observation):
        return observation['image']