# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:58:32 2022

@author: Ghiglione
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns

plt.style.use('seaborn')


##################################################################
# Read the data
df = pd.read_csv('data\london_merged.csv')



##################################################################
# Obtain time info from timestamp
df['weekday'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
df['year'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
df['month'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
df['hour'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
df = df.drop(columns=['timestamp'])

print(df.describe())



##################################################################
# Barplot of continuous vs discrete variable
fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=df['weekday'], y=df['cnt'], color='royalblue', ax=ax)
plt.tight_layout()



###################################################################
# Histogram of continuos variable w.r.t discrete variable values
g = sns.FacetGrid(df, col='season', margin_titles=True)
g = g.map(plt.hist, 'cnt')


g = sns.FacetGrid(df, row='season', col='weekday', margin_titles=True)
g = g.map(plt.hist, 'cnt')














