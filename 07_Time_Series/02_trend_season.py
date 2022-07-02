# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 08:48:06 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from scipy.stats import norm

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


###################################################################################
# read data
df = pd.read_csv('data/others/a10.csv', parse_dates=['date'])

# Extract info from timestamp
df['weekday'] = df['date'].apply(lambda x: x.weekday())
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)

df = df.set_index('date', drop=True)

fig, ax = plt.subplots(1,1)
#ax.plot(df['date'], df['value'])
ax.plot(df['value'])
ax.set_title('Time Series')


fig, ax = plt.subplots(1,3, sharey=(True), figsize=(15,5))
h = sns.boxplot(x='year', y='value', data=df, ax=ax[0], color='royalblue')
h.set_xticklabels(h.get_xticklabels(),rotation=90)
sns.boxplot(x='month', y='value', data=df, ax=ax[1], color='royalblue')
sns.boxplot(x='weekday', y='value', data=df, ax=ax[2], color='royalblue')
fig.tight_layout()


# fig, ax = plt.subplots(1,3, sharey=(True), figsize=(10,5))
# df[['value', 'year']].boxplot(by='year', ax=ax[0])
# plt.xticks(rotation=90)
# df[['value', 'month']].boxplot(by='month', ax=ax[1])
# df[['value', 'weekday']].boxplot(by='weekday', ax=ax[2])
# fig.tight_layout()


####################################################################################
# Decompose trend and seasonalities
# function needs a pandas object with indexes the date (with no missing values)
decomp_mult = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
decomp_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')


fig = decomp_mult.plot()
fig.suptitle('Multiplicative Decomposition')     # Y[t] = T[t] * S[t] * e[t]
fig.tight_layout()

fig = decomp_add.plot()
fig.suptitle('Additive Decomposition')           # Y[t] = T[t] + S[t] + e[t]
fig.tight_layout()

# decomp_mult.trend   decomp_mult.seasonal   decomp_mult.resid

#############################
# set period by hand
# decomp_mult_h = seasonal_decompose(df['value'], model='multiplicative', period=30) # 
# decomp_add_h = seasonal_decompose(df['value'], model='additive', period=30)


# fig = decomp_mult_h.plot()
# fig.suptitle('Multiplicative Decomposition')     # Y[t] = T[t] * S[t] * e[t]
# fig.tight_layout()

# fig = decomp_add_h.plot()
# fig.suptitle('Additive Decomposition')           # Y[t] = T[t] + S[t] + e[t]
# fig.tight_layout()


#################################################################################
# Residuals Diagnostic
res = decomp_mult.resid
res_std = (res - np.mean(res))/np.std(res)

fig, ax = plt.subplots(2,2)
fig.suptitle('Diagnostic')

ax[0,0].plot(res_std)
ax[0,0].set_title('Residuals')

plot_acf(res_std, ax=ax[0,1])

sns.distplot(res_std, bins=15, kde=True, ax=ax[1,0], label='Std Resid')
xx = np.arange(-4, 4, 0.01)
ax[1,0].plot(xx, norm.pdf(xx, 0, 1), label='N(0,1)')
ax[1,0].set_title('Std Residuals Histogram')
ax[1,0].legend()

sm.qqplot(res_std, line='45', fit=True, ax=ax[1,1])
ax[1,1].set_title('QQPlot of Residuals')

fig.tight_layout()

test_dw = durbin_watson(res_std)  # needs std residuals, values around 2 mean no serial correlation
print(f'Durbin-Watson test for serial correlation: {test_dw}')









