# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:39:37 2023

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
import statsmodels.api as sm

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


####################################################################################
# Huber Regression
from sklearn.linear_model import HuberRegressor

eps = 1.90        # [1,inf), the lower, the higher is the robust effect
alph = 0.0001     # L2 regularization
mod = HuberRegressor(epsilon=eps, alpha=alph)

mod.fit(x, y)

y_hat = mod.predict(x)

r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
rss = sum( (y_hat-y)**2 )
mse = np.mean( (y - y_hat)**2 )

# Plot
x_range = np.max(x)-np.min(x)
a = 0.05
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
yy = mod.predict(xx)
XX = sm.add_constant(xx)
conf = 0.95
alp = 1 - conf
tq = stats.t(df=(n-r-1)).ppf(1-alp/2)
IC_len = tq * np.sqrt( mse * ( 1 + np.diag( XX @ np.linalg.inv(XX.T @ XX) @ XX.T) ) )


fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.scatter(x[mod.outliers_], y[mod.outliers_], color='red', label='outliers') # non sono outliers, ma i dati dove decade la Huber function
ax.plot(xx, yy, color='red')
ax.fill_between(xx.reshape(-1,), yy - IC_len, yy + IC_len, color='red', alpha=0.2)
ax.set_title(f'Fitted Model, eps={eps}')
ax.legend()