# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 09:43:55 2022

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

df = pd.read_csv('data/others/hospital.csv',index_col='date', parse_dates=['date'])
df = df.rename(columns={'wait_times_hrs':'y', 'nurse_count':'x'})

fig, ax = plt.subplots(1,1)
ax.plot(df['y'])


######################################################################################
# Test Stationarity (Augmented Dickey-Fuller Test)
# H0: the series is random walk => non stationary
# H1: the series is not random walk => may be stationary

adf = adfuller(df['y'], regression='c')[1]
print(f'Adf Test: pvalue={adf}')
# I reject H0, the process may be stationary


######################################################################################
# ARMA model selection
fig, ax = plt.subplots(1,2)
plot_acf(df['y'], ax=ax[0])    # cuts if process is MA(q)
plot_pacf(df['y'], ax=ax[1])   # cuts if process is AR(q)
fig.tight_layout()
# they both don't cut, I have an ARMA

p = []
q = []
bic = []
aic = []
ar_max = 5
ma_max = 5

for ar_order in tqdm(range(ar_max)):
    for ma_order in range(ma_max):
        model = ARIMA(df['y'], order=(ar_order, 0, ma_order))
        result = model.fit()
        
        p.append(ar_order)
        q.append(ma_order)
        bic.append(result.bic)
        aic.append(result.aic)
        
gridsearch = pd.DataFrame({'AR(p)':p,'MA(q)':q,'AIC':aic,'BIC':bic})

aic_selected = gridsearch.sort_values(by='AIC', ascending=True).iloc[:4,:]
bic_selected = gridsearch.sort_values(by='BIC', ascending=True).iloc[:4,:]

print('\nAIC Selected')
print(aic_selected)
print('\nBIC Selected')
print(bic_selected)


###################################################################################
# Fit ARMA model
mod_arma = ARIMA(df['y'], order=(2,0,4)).fit()

y_hat = mod_arma.predict(df['y'])
fig, ax = plt.subplots(1,1)
ax[0].plot(df['y'])
ax[1].plot(y_hat)

# diagnostic
mod_arma.plot_diagnostics(ax=ax)





















