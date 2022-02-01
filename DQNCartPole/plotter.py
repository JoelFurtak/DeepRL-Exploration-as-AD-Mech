from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_style(style='darkgrid')

dfs = []

for name in ['decay_97', 'decay_95', '3k_random_steps']:
    for i in range(4):
        try:                                                        # i = 1..3
            df = pd.read_csv('./data/{}/run#{}/results.csv'.format(name, i), index_col=False)
            df.score = df.score.rolling(100).mean()
            df['Parameters'] = name
            dfs.append(df)
        except Exception as e:                                      # i = 0
            pass
    

df = pd.concat(dfs).reset_index()

sns.lineplot(data=df, x='episode', y='score', hue='Parameters' , palette='husl', ci=95, n_boot=50000)
plt.show()