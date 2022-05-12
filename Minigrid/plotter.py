import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math

sns.set_style(style='darkgrid')

#utils.plot_scores('ppo', 'dk6x6', 0)
#utils.plot_average('ppo', 'dk6x6', 0)

dfs = []
for name in ['dqn', 'ppo', 'ppo_rnd']:
    for i in range(6):
        try:
            df = pd.read_csv(f'./data/{name}/dk6x6/run#{i}/results.csv', index_col=False)
            df.score = df.collisions.rolling(100).mean()
            df['Parameters'] = name
            dfs.append(df)
        except Exception as e:                                      # i = 0
            pass
    

df = pd.concat(dfs).reset_index()

sns.lineplot(data=df, x='episode', y='score', hue='Parameters' , palette='husl', ci='sd')
plt.show()