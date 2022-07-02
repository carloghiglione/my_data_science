# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 16:22:36 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


##################################################################################
# Read data
x1 = pd.read_csv('data/stocks/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
x2 = pd.read_csv('data/stocks/AMZN_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
x3 = pd.read_csv('data/stocks/MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
x4 = pd.read_csv('data/stocks/BA_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

df = x1['Close'].to_frame('X1')
df = df.join(x2['Close'].to_frame('X2'))
df = df.join(x3['Close'].to_frame('X3'))
df = df.join(x4['Close'].to_frame('X4'))

df_t = pd.DataFrame({'date': df.index})
df_t['weekday'] = df_t['date'].apply(lambda x: x.weekday())
df_t['year'] = df_t['date'].apply(lambda x: x.year)
df_t['month'] = df_t['date'].apply(lambda x: x.month)

fig, ax = plt.subplots(1,1)
#ax.plot(df['date'], df['value'])
df.plot(ax=ax)
ax.set_title('Time Series')


##################################################################################
# Study correlation among the components

corrmat = df.corr()
fig, ax = plt.subplots(1,1)
sns.heatmap(corrmat, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Correlation')
plt.tight_layout()


g = sns.PairGrid(df, palette=["red"], diag_sharey=False) 
g.map_upper(sns.regplot, line_kws={'color':'red'}) 
g.map_diag(sns.distplot, kde=True) 
g.map_lower(sns.regplot, line_kws={'color':'red'}) 


###################################################################################
# Study correlation among the percentage increase

df_perc_incr = df.pct_change()

fig, ax = plt.subplots(1,1)
df_perc_incr.plot(ax=ax)
ax.set_title('Percentage Increase')


corrmat = df_perc_incr.corr()
fig, ax = plt.subplots(1,1)
sns.heatmap(corrmat, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Correlation Percentage Increment')
plt.tight_layout()


g = sns.PairGrid(df_perc_incr, palette=["red"], diag_sharey=False) 
g.map_upper(sns.regplot, line_kws={'color':'red'}) 
g.map_diag(sns.distplot, kde=True) 
g.map_lower(sns.regplot, line_kws={'color':'red'}) 







