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
# # Read the data
# df = pd.read_csv('data\london_merged.csv')

# # Obtain time info from timestamp
# df['weekday'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
# df['year'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
# df['month'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
# df['hour'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
# df = df.drop(columns=['timestamp'])

# print(df.describe())


####################################################################
# Read the data (already parse timestamp)
df = pd.read_csv('data\london_merged.csv', parse_dates=['timestamp'])

# Obtain time info from timestamp
df['weekday'] = df['timestamp'].apply(lambda x: x.weekday())
df['year'] = df['timestamp'].apply(lambda x: x.year)
df['month'] = df['timestamp'].apply(lambda x: x.month)
df['hour'] = df['timestamp'].apply(lambda x: x.hour)
df['minute'] = df['timestamp'].apply(lambda x: x.minute)
df['second'] = df['timestamp'].apply(lambda x: x.second)

df = df.set_index('timestamp', drop=True)


##################################################################
# Barplot of continuous vs discrete variable
fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=df['weekday'], y=df['cnt'], color='royalblue', ax=ax)
plt.tight_layout()


###################################################################
# Histogram of continuos variable w.r.t discrete variable values
# g = sns.FacetGrid(df, col='season', margin_titles=True)
# g = g.map(plt.hist, 'cnt')


# g = sns.FacetGrid(df, row='season', col='weekday', margin_titles=True)
# g = g.map(plt.hist, 'cnt')



####################################################################
# Subrange of Time Series
df_interval = df['2015' : '2016']
df_interval = df['2015-01' : '2016-03']
df_interval = df['2015-01-02' : '2016-03-23']
df_interval = df['2015-01-02 19' : '2016-01-07 18']


####################################################################
# Subset of Time Series
df_sub = df.resample('M').sum()     # 'D' day, '3D' 3 days, 'W' week, 'M' month, 'Y' year 
                                    # sum(), max(), min(), mean()
                            
####################################################################
# Refill the time seriesin missing times
df_refill = df_sub.asfreq(freq='3D', method='bfill') # 'bfill' last value, 'fflil' next value


####################################################################
# Step and Moving Average
df_day = df.resample('D').mean()

df_step_av = df_day['hum'].resample('M').mean()
df_m_av = df_day['hum'].rolling(30).mean()

fig, ax = plt.subplots(1,1)
ax.plot(df_day['hum'], label='data')
ax.plot(df_step_av, label='step av')
ax.plot(df_m_av, label='moving av')
ax.legend()
fig.tight_layout()


#####################################################################
# Shift, Step Difference and Percentage Change
df_shift = df.shift(periods = 1)  # forward shift, if negative can be negative shift
df_diff = df.diff()               # can set the period
df_perc_change = df.pct_change()  # can set the period

















