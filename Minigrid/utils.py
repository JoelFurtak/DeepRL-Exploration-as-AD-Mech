import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math

#sns.set_style(style='darkgrid')

def save_data(episodes, scores, collisions, pick_ups, drops, toggles, key_pickups, key_drops, door_toggles, turns, intrinsic_reward, alg, short_name, run):
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

    print('... saving data ...')
    df = {'episode': episodes, 'score': scores, 'collisions': collisions, 'pick_ups': pick_ups, 'drops': drops, 'toggles': toggles, 'key_pickups': key_pickups, 'key_drops': key_drops, 'door_toggles':door_toggles, 'turns':turns, 'intrinsic_reward':intrinsic_reward}
    df = pd.DataFrame(df)
    df.to_csv('./data/{}/{}/run#{}/results.csv'.format(alg, short_name, run + 1))

def plot_scores(alg, short_name, run):
    df = pd.read_csv('./data/{}/{}/run#{}/results.csv'.format(alg, short_name, run + 1))

    sns.lineplot(data=df, x='episode', y='score')
    plt.show()

def plot_average(alg, short_name, run):
    df = pd.read_csv('./data/{}/{}/run#{}/results.csv'.format(alg, short_name, run + 1))
    df.score = df.score.rolling(100).mean()

    sns.lineplot(data=df, x='episode', y='score')
    plt.show()

def plot_mlt_average(alg, short_names):
    dfs = []
    for name in short_names:
        for i in range(6):
            try:                                                        # i = 1..5
                df = pd.read_csv('./data/{}/{}/run#{}/results.csv'.format(alg, name, i), index_col=False)
                df.score = df.score.rolling(100).mean()
                df['Parameters'] = name
                dfs.append(df)
            except Exception as e:                                      # i = 0
                pass
    

    df = pd.concat(dfs).reset_index()

    sns.lineplot(data=df, x='episode', y='score', hue='Parameters' , palette='husl', ci='sd')
    plt.show()

