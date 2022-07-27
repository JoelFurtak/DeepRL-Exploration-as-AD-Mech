from turtle import title
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set(font='Arial')
sns.set_style(style='ticks')

def evaluate(alg, env):
    dfs = []
    for name in [alg]:
        for i in range(6):
            try:
                df = pd.read_csv(f'./data/{name}/{env}/run#{i}/results_train.csv', index_col=False)
                df.score = df.score.rolling(100, min_periods=100).mean()
                df['Legende'] = 'Trainierender Agent'
                dfs.append(df)
            except Exception as e:                                      # i = 0
                pass
    
    for name in [alg]:
        for i in range(6):
            try:
                df = pd.read_csv(f'./data/{name}/{env}/run#{i}/results_play.csv', index_col=False)
                df.score = df.score.rolling(100, min_periods=100).mean()
                df['Legende'] = 'Eingesetzter Agent'
                dfs.append(df)
            except Exception as e:                                      # i = 0
                pass

    df = pd.concat(dfs).reset_index()

    sns.lineplot(data=df, x='episode', y='score', hue='Legende' , palette='mako', ci='sd').set(title='Trainierender Agent vs Eingesetzter Agent in Lava Gap: \nBelohnungen (Durchschnitt)', xlabel='Episode', ylabel='Belohnung')
    plt.legend(title=None, loc='lower right', frameon=False)
    plt.show()

evaluate(alg='ppo', env='lava')