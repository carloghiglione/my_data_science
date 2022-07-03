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

train_split = 0.80
y = df['y'].values
X = df['x'].values

n = len(y)
n_train = int(n*train_split)
n_test = n - n_train
y_train, y_test = y[:n_train], y[n_train:]
X_train, X_test = X[:n_train], X[n_train:]
t_train, t_test = df.index[:n_train], df.index[n_train:]

fig, ax = plt.subplots(1,1)
ax.plot(t_train, y_train, label='Train')
ax.plot(t_test, y_test, label='Test')
ax.legend()
ax.set_title('Data')
plt.tight_layout()

######################################################################################
# Test Stationarity (Augmented Dickey-Fuller Test)
# H0: the series is random walk => non stationary
# H1: the series is not random walk => may be stationary

adf = adfuller(y_train, regression='c')[1]
print(f'Adf Test: pvalue={adf}')
# I reject H0, the process may be stationary


######################################################################################
# ARMA model selection
fig, ax = plt.subplots(1,2)
plot_acf(y_train, ax=ax[0])    # cuts if process is MA(q)
plot_pacf(y_train, ax=ax[1])   # cuts if process is AR(q)
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
        model = ARIMA(y_train, order=(ar_order, 0, ma_order))
        result = model.fit()
        
        p.append(ar_order)
        q.append(ma_order)
        bic.append(result.bic)
        aic.append(result.aic)
        
gridsearch = pd.DataFrame({'AR(p)':p,'MA(q)':q,'AIC':aic,'BIC':bic})

aic_selected = gridsearch.sort_values(by='AIC', ascending=True).iloc[:4,:]
bic_selected = gridsearch.sort_values(by='BIC', ascending=True).iloc[:4,:]

print('\nARMA: AIC Selected')
print(aic_selected)
print('\nARMA: BIC Selected')
print(bic_selected)


###################################################################################
# Fit ARMA model
ar_best = 2
ma_best = 4

mod_arma = ARIMA(y_train, order=(ar_best, 0, ma_best)).fit()

mod_pred = mod_arma.get_prediction()
y_hat_train = mod_pred.predicted_mean
y_hat_conf = mod_pred.conf_int()

fig, ax = plt.subplots(1,1)
ax.plot(t_train, y_train, label='Data')
ax.plot(t_train, y_hat_train, color='red', label='Fitted')
ax.fill_between(t_train, y_hat_conf[:,0], y_hat_conf[:,1], 
                color='red', alpha=0.25, label='Conf 95%')
ax.legend()
ax.set_title(f'ARMA({ar_best}, {ma_best}), Model')

########################
# diagnostic
print(mod_arma.summary())

mod_arma.plot_diagnostics()
plt.tight_layout()
print(f'Durbin-Watson Test: {durbin_watson(mod_arma.resid)}')


########################
# Prediction
y_hat_list = []
res_list = []
for i in range(n_test):
    y_hat = mod_arma.forecast(steps=1)
    res = y_test[i] - y_hat
    y_hat_list.append(y_hat)
    res_list.append(res)
    mod_arma = mod_arma.append([y_test[i]], refit=False)

res_list = np.array(res_list)
rmse_test_arma = np.sqrt(np.mean(res_list**2))
print(f'RMSE ARMA Test: {rmse_test_arma}')

fig, ax = plt.subplots(1,1)
ax.plot(y_test, label='data')
ax.plot(y_hat_list, label='prediction')
ax.legend()
ax.set_title('ARMA Prediction')


mod_pred = mod_arma.get_prediction()
y_hat= mod_pred.predicted_mean
y_hat_conf = mod_pred.conf_int()

fig, ax = plt.subplots(1,1)
ax.plot(df.index, y, label='Data')
ax.plot(df.index, y_hat, color='red', label='Fitted')
ax.fill_between(df.index, y_hat_conf[:,0], y_hat_conf[:,1], 
                color='red', alpha=0.25, label='Conf 95%')
ax.axvline(df.index[n_train], color='black', ls='--')
ax.set_title('ARMA Model & Prediction')



###################################################################################
# ARMAX model selection
from sklearn.preprocessing import PolynomialFeatures
# polynomial = PolynomialFeatures(degree=3, include_bias=False)
# x = polynomial.fit_transform(x.reshape(-1,1))

p = []
q = []
bic = []
aic = []
ar_max = 5
ma_max = 5

for ar_order in tqdm(range(ar_max)):
    for ma_order in range(ma_max):
        model = ARIMA(y_train, order=(ar_order, 0, ma_order), exog=X_train)
        result = model.fit()
        
        p.append(ar_order)
        q.append(ma_order)
        bic.append(result.bic)
        aic.append(result.aic)
        
gridsearch = pd.DataFrame({'AR(p)':p,'MA(q)':q,'AIC':aic,'BIC':bic})

aic_selected = gridsearch.sort_values(by='AIC', ascending=True).iloc[:4,:]
bic_selected = gridsearch.sort_values(by='BIC', ascending=True).iloc[:4,:]

print('\nARMAX AIC Selected')
print(aic_selected)
print('\nARMAX BIC Selected')
print(bic_selected)


###################################################################################
# Fit ARMAX model
ar_best = 1
ma_best = 1
mod_armax = ARIMA(y_train, order=(ar_best, 0, ma_best), exog=X_train).fit()

mod_pred = mod_armax.get_prediction()
y_hat = mod_pred.predicted_mean
y_hat_conf = mod_pred.conf_int()

fig, ax = plt.subplots(1,1)
ax.plot(t_train, y_train, label='Data')
ax.plot(t_train, y_hat, color='red', label='Fitted')
ax.fill_between(t_train, y_hat_conf[:,0], y_hat_conf[:,1], 
                color='red', alpha=0.25, label='Conf 95%')
ax.legend()
ax.set_title(f'ARMAX({ar_best}, {ma_best}), Model')

####################
# diagnostic
print(mod_armax.summary())

mod_armax.plot_diagnostics()
plt.tight_layout()
print(f'Durbin-Watson Test: {durbin_watson(mod_arma.resid)}')


########################
# Prediction
y_hat_list = []
res_list = []
for i in range(n_test):
    y_hat = mod_armax.forecast(steps=1, exog=X_test[i])
    res = y_test[i] - y_hat
    y_hat_list.append(y_hat)
    res_list.append(res)
    mod_armax = mod_armax.append([y_test[i]], exog=[X_test[i]], refit=False)

res_list = np.array(res_list)
rmse_test_armax = np.sqrt(np.mean(res_list**2))
print(f'RMSE ARMAX Test: {rmse_test_arma}')

fig, ax = plt.subplots(1,1)
ax.plot(y_test, label='data')
ax.plot(y_hat_list, label='prediction')
ax.legend()
ax.set_title('ARMAX Prediction')


mod_pred = mod_armax.get_prediction()
y_hat= mod_pred.predicted_mean
y_hat_conf = mod_pred.conf_int()

fig, ax = plt.subplots(1,1)
ax.plot(df.index, y, label='Data')
ax.plot(df.index, y_hat, color='red', label='Fitted')
ax.fill_between(df.index, y_hat_conf[:,0], y_hat_conf[:,1], 
                color='red', alpha=0.25, label='Conf 95%')
ax.axvline(df.index[n_train], color='black', ls='--')
ax.set_title('ARMAX Model & Prediction')


##################################################################################
# Model Comparison
arma_rmse = np.sqrt(np.mean(mod_arma.resid**2))
armax_rmse = np.sqrt(np.mean(mod_armax.resid**2))

print(f'ARMA RMSE: {arma_rmse}')
print(f'ARMAX RMSE: {armax_rmse}')

