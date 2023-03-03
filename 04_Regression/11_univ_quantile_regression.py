# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:37:35 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from scipy import stats

plt.style.use('seaborn')

 
####################################################################################
# Read the data

df = pd.read_csv('data/house_price_train.csv', index_col=0)
y = df['SalePrice']
x = df['YearBuilt'].values.reshape(-1,1)


fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.set_title('Data')

n = len(x)
r = 1


###################################################################################
# Perform Quantile Regression (prediction is the median)
from sklearn.utils.fixes import sp_version, parse_version

# This is line is to avoid incompatibility if older SciPy version.
# You should use `solver="highs"` with recent version of SciPy.
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

from sklearn.linear_model import QuantileRegressor

alpha = 0.05
quantiles = [alpha, 0.5, 1-alpha]
quant_names = ['low', 'med', 'up']
predictions = {}

for i in range(3):
    qr = QuantileRegressor(quantile=quantiles[i], alpha=0, solver=solver)  # set alpha for L1 regularization
    y_pred = qr.fit(x, y).predict(x)
    predictions[quant_names[i]] = y_pred
y_hat = predictions['med']  

# plot
x_range = np.max(x)-np.min(x)
a = 0.05
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
pred_xx = {}

for i in range(3):
    qr = QuantileRegressor(quantile=quantiles[i], alpha=0, solver=solver)
    yy_pred = qr.fit(x, y).predict(xx)
    pred_xx[quant_names[i]] = yy_pred

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(xx, pred_xx['med'], color='red', label='median')
ax.fill_between(xx.reshape(-1,), pred_xx['low'], pred_xx['up'], color='red', alpha=0.2, label=f'{100*(1-alpha)}% quantile interval')
ax.set_title('Quantile Regression')
ax.legend()