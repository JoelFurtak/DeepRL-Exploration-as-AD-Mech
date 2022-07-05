import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set_style(style='darkgrid')

def evaluate(alg, env):
    dfs = []
    for name in [alg]:
        for i in range(6):
            try:
                df = pd.read_csv(f'./data/{name}/{env}/run#{i}/results_train.csv', index_col=False)
                df.key_pickups = df.key_pickups.rolling(100, min_periods=100).mean()
                df['Parameters'] = 'Trainee'
                dfs.append(df)
            except Exception as e:                                      # i = 0
                pass
        
    for name in [alg]:
        for i in range(6):
            try:
                df = pd.read_csv(f'./data/{name}/{env}/run#{i}/results_play.csv', index_col=False)
                df.key_pickups = df.key_pickups.rolling(100, min_periods=100).mean()
                df['Parameters'] = 'Player'
                dfs.append(df)
            except Exception as e:                                      # i = 0
                pass

    df = pd.concat(dfs).reset_index()

    sns.lineplot(data=df, x='episode', y='key_pickups', hue='Parameters' , palette='husl', ci='sd')
    plt.show()

evaluate(alg='ppo', env='dk6x6')