# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:29:54 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy.stats import norm
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


######################################################################################
# Read Data
df = pd.read_csv('data/stocks/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
df['Var'] = df['High'] - df['Low']
df = df.drop(columns=['Name', 'High', 'Low'])
y_s = df['Close']
X_df = df.loc[:, df.columns != 'Close']



#######################################################################################
# Test Stationarity (Augmented Dickey-Fuller Test)
# H0: the series is random walk => non stationary
# H1: the series is not random walk => may be stationary

adf = adfuller(y_s.values, regression='c')[1]
print(f'Adf Test: pvalue={adf}')
# I don't reject H0, the process is not stationary


#######################################################################################
# ARIMA Model Selection
y = y_s.values

p = []
q = []
r = []
bic = []
aic = []
ar_max = 3
ma_max = 3
r_max = 3

for ar_order in tqdm(range(ar_max)):
    for ma_order in range(ma_max):
        for r_order in range(r_max):
            model = ARIMA(y, order=(ar_order, r_order, ma_order))
            result = model.fit()
        
            p.append(ar_order)
            q.append(ma_order)
            r.append(r_order)
            bic.append(result.bic)
            aic.append(result.aic)

gridsearch = pd.DataFrame({'AR(p)':p,'MA(q)':q,'I(r)':r,'AIC':aic,'BIC':bic})

aic_selected = gridsearch.sort_values(by='AIC', ascending=True).iloc[:4,:]
bic_selected = gridsearch.sort_values(by='BIC', ascending=True).iloc[:4,:]

print('\nARIMA: AIC Selected')
print(aic_selected)
print('\nARIMA: BIC Selected')
print(bic_selected)



###################################################################################
# ARIMA Model Fitting
ar_best = 2
ma_best = 2
r_best = 1
mod_arima = ARIMA(y, order=(ar_best, r_best, ma_best)).fit()

mod_pred = mod_arima.get_prediction()
y_hat = mod_pred.predicted_mean
y_hat_conf = mod_pred.conf_int()

fig, ax = plt.subplots(1,1)
ax.plot(df.index, y, label='Data')
ax.plot(df.index, y_hat, color='red', label='Fitted')
ax.fill_between(df.index[20:], y_hat_conf[20:,0], y_hat_conf[20:,1], 
                color='red', alpha=0.25, label='Conf 95%')
ax.legend()
ax.set_title(f'ARIMA({ar_best}, {r_best}, {ma_best}), Model')
fig.tight_layout()

# diagnostic
print(mod_arima.summary())

mod_arima.plot_diagnostics()
plt.tight_layout()
print(f'Durbin-Watson Test: {durbin_watson(mod_arima.resid)}')



#######################################################################################
# ARIMAX Model Selection
y = y_s.values
X = X_df.values

p = []
q = []
r = []
bic = []
aic = []
ar_max = 3
ma_max = 3
r_max = 3

for ar_order in tqdm(range(ar_max)):
    for ma_order in range(ma_max):
        for r_order in range(r_max):
            model = ARIMA(y, order=(ar_order, r_order, ma_order), exog=X)
            result = model.fit()
        
            p.append(ar_order)
            q.append(ma_order)
            r.append(r_order)
            bic.append(result.bic)
            aic.append(result.aic)

gridsearch = pd.DataFrame({'AR(p)':p,'MA(q)':q,'I(r)':r,'AIC':aic,'BIC':bic})

aic_selected = gridsearch.sort_values(by='AIC', ascending=True).iloc[:4,:]
bic_selected = gridsearch.sort_values(by='BIC', ascending=True).iloc[:4,:]

print('\nARIMAX: AIC Selected')
print(aic_selected)
print('\nARIMAX: BIC Selected')
print(bic_selected)



###################################################################################
# ARIMAX Model Fitting
ar_best = 0
ma_best = 1
r_best = 0
mod_arimax = ARIMA(y, order=(ar_best, r_best, ma_best), exog=X).fit()

mod_pred = mod_arimax.get_prediction()
y_hat = mod_pred.predicted_mean
y_hat_conf = mod_pred.conf_int()

fig, ax = plt.subplots(1,1)
ax.plot(df.index, y, label='Data')
ax.plot(df.index, y_hat, color='red', label='Fitted')
ax.fill_between(df.index[20:], y_hat_conf[20:,0], y_hat_conf[20:,1], 
                color='red', alpha=0.25, label='Conf 95%')
ax.legend()
ax.set_title(f'ARIMAX({ar_best}, {r_best}, {ma_best}), Model')
fig.tight_layout()

# diagnostic
print(mod_arimax.summary())

mod_arimax.plot_diagnostics()
plt.tight_layout()
print(f'Durbin-Watson Test: {durbin_watson(mod_arimax.resid)}')








