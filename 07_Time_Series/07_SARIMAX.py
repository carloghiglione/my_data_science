# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 14:37:14 2022

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
from statsmodels.tsa.seasonal import seasonal_decompose


plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


######################################################################################
# Read Data
df = pd.read_csv('data/others/milk_production.csv', index_col='Month', parse_dates=['Month'])
df = df.rename(columns = {'pounds_per_cow': 'y'})
y_s = df['y']

df = pd.read_csv('data/others/a10.csv', parse_dates=['date'], index_col='date')
y_s = df['value']

fig, ax = plt.subplots(1,1)
ax.plot(y_s)
fig.tight_layout()


####################################################################################
# Test Stationarity (Augmented Dickey-Fuller Test)
# H0: the series is random walk => non stationary
# H1: the series is not random walk => may be stationary

adf = adfuller(y_s.values, regression='c')[1]
print(f'Adf Test: pvalue={adf}')
# I don't reject H0, the process is not stationary


#####################################################################################
# Seasonal trend extraction
decomp_mult = seasonal_decompose(y_s, model='multiplicative', extrapolate_trend='freq')

fig = decomp_mult.plot()
fig.suptitle('Multiplicative Decomposition')     # Y[t] = T[t] * S[t] * e[t]
fig.tight_layout()

plot_acf(decomp_mult.seasonal)
p_best = 12


#####################################################################################
# SARIMA Model Selection
y = y_s.values

p = []
q = []
r = []
bic = []
aic = []
ar_max = 3
ma_max = 3
r_max = 2

for ar_order in tqdm(range(ar_max)):
    for ma_order in range(ma_max):
        for r_order in range(r_max):
            model = ARIMA(y, order=(ar_order, r_order, ma_order), seasonal_order=(1,1,1,p_best))
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



#################################################################################
# Fit SARIMA Process
ar_best = 1
ma_best = 1
r_best = 1
mod_arima = ARIMA(y, order=(ar_best, r_best, ma_best), seasonal_order=(1,1,1,p_best)).fit()

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












