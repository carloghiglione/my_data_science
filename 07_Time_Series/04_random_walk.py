# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:08:04 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from scipy.stats import norm
from tqdm import tqdm

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


##################################################################################
# Read data
df = pd.read_csv('data/stocks/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

x1 = df['Volume']
x2 = df['Close']

fig, ax = plt.subplots(1,2)
ax[0].plot(x1)
ax[0].set_title('X1')
ax[1].plot(x2)
ax[1].set_title('X2')
fig.tight_layout()


####################################################################################
# Augmented Fuller Test for Random Walks with Drift
# H0: the series is random walk => non stationary
# H1: the series is not random walk => may be stationary

np.random.seed(1234)
wn = 10*np.random.normal(size=200)             # Y(t) = e(t)                    white noise                       
rw = np.cumsum(wn)                             # Y(t) = Y(t-1) + e(t)           random walk
rw_c = np.cumsum(wn + 1)                       # Y(t) = a + Y(t-1) + e(t)       random walk with constant drift
rw_l= np.cumsum(wn + 0.01*np.arange(0,200))    # Y(t) = a + bt + Y(t-1) + e(t)  random walk with constant drift and linear trend
rw_q= np.cumsum(wn + 0.01*np.arange(0,200) - 0.0001*np.arange(0,200)**2)  # Y(t) = a + bt + ct^2 + Y(t-1) + e(t)  random walk with constant drift and linear and quadratic trend

xx = rw
fig, ax = plt.subplots(1,1)
ax.plot(xx)
adfuller(xx, regression='c')[1]  # 'c':constant drift, 'ct':constant drift, linear trend, 'ctt':constant drift, linear and quadratic trend


adf1 = adfuller(x1, regression='c')[1]
adf2 = adfuller(x2, regression='c')[1]

print(f'X1: ADF test, pvalue={adf1}')
print(f'X2: ADF test, pvalue={adf2}')


####################################################################################
# Extract the trend
x2_diff = x2.diff().dropna()
x2_trend_mean = np.mean(x2_diff)

# visualize residuals (a + e(t))
fig, ax = plt.subplots(1,1)
ax.plot(x2_diff)


res = x2_diff.values
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


######################################################################################
# perform a nonparametric test to test the value of the trend
# H0: trend = 0
# H1: trend != 0

def statistic(x, mu0):
    return np.abs(np.mean(x)-mu0)

def permute(x, mu0):
    return mu0 + (x.copy() - mu0)*np.random.choice(np.array([-1,1]), size=len(x))

T_list = []
mu0 = 0
T0 = statistic(res, mu0)
B = 10000
for b in tqdm(range(B)):
    x_perm = permute(res, mu0)
    T_perm = statistic(x_perm, mu0)
    T_list.append(T_perm)

fig, ax = plt.subplots(1,1, figsize=(8,4))
sns.distplot(T_list, bins=30, kde=True, ax=ax)
ax.axvline(T0, color='red')
ax.set_title('T perm')
plt.tight_layout()

pvalue = np.sum(T_list > T0)/B
print(f'Test, H0: drift=0 | pvalue={pvalue}')













