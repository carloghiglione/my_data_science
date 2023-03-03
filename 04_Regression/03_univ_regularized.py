# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 07:59:53 2022

@author: Utente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from scipy import stats

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings('ignore')

####################################################################################
# Read the data
df = pd.read_csv('data/house_price_train.csv', index_col=0)
y = df['SalePrice']
x = df['YearBuilt'].values.reshape(-1,1)

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.set_title('Data')

n = len(x)



######################################################################################
# Ridge Regression (Sklearn)
from sklearn.linear_model import Ridge
r = 5
polynomial = PolynomialFeatures(degree=r, include_bias=False)
X_poly = polynomial.fit_transform(x)

alpha = 0.1
mod_ridge = Ridge(alpha=alpha, random_state=1234)
mod_ridge.fit(X_poly, y)

y_hat = mod_ridge.predict(X_poly)
r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
rmse = np.sqrt(np.mean( (y - y_hat)**2))
mse = np.mean( (y - y_hat)**2 )

# Plot
x_range = np.max(x)-np.min(x)
a = 0.05
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
yy = mod_ridge.predict(polynomial.fit_transform(xx))
XX = sm.add_constant(polynomial.fit_transform(xx))
conf = 0.95
alp = 1 - conf
tq = stats.t(df=(n-r-1)).ppf(1-alp/2)
IC_len = tq * np.sqrt( mse * ( 1 + np.diag( XX @ np.linalg.inv(XX.T @ XX) @ XX.T) ) )


fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(xx, yy, color='red')
ax.fill_between(xx.reshape(-1,), yy - IC_len, yy + IC_len, color='red', alpha=0.2)
ax.set_title('Fitted Model, Ridge')


######################################################################################
# Ridge Regression - Parameter CV (Sklearn)
from sklearn.linear_model import RidgeCV

a_grid = np.arange(1e-3, 5, 1e-3)
ridge_cv = RidgeCV(alphas=a_grid, store_cv_values=True).fit(X_poly,y)
alpha_best = ridge_cv.alpha_
mse_cv = np.mean(ridge_cv.cv_values_, axis=0)

fig, ax = plt.subplots(1,1)
ax.plot(a_grid, mse_cv)
ax.axvline(alpha_best, color='red')
ax.set_title('Ridge CV, alpha selection')


#######################################################################################
# LASSO Regression (Sklearn)
from sklearn.linear_model import Lasso

alpha = 0.000576
mod_lasso = Lasso(alpha=alpha, random_state=1234)
mod_lasso.fit(X_poly, y)

y_hat = mod_lasso.predict(X_poly)
y_hat = mod_ridge.predict(X_poly)
r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
rmse = np.sqrt(np.mean( (y - y_hat)**2))
mse = np.mean( (y - y_hat)**2 )

# Plot
x_range = np.max(x)-np.min(x)
a = 0.05
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
yy = mod_lasso.predict(polynomial.fit_transform(xx))
XX = sm.add_constant(polynomial.fit_transform(xx))
conf = 0.95
alp = 1 - conf
tq = stats.t(df=(n-r-1)).ppf(1-alp/2)
IC_len = tq * np.sqrt( mse * ( 1 + np.diag( XX @ np.linalg.inv(XX.T @ XX) @ XX.T) ) )

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(xx, yy, color='red')
ax.fill_between(xx.reshape(-1,), yy - IC_len, yy + IC_len, color='red', alpha=0.2)
ax.set_title('Fitted Model, Lasso')


######################################################################################
# Lasso Regression - Parameter CV (Sklearn)
from sklearn.linear_model import LassoCV

a_grid = np.arange(1e-6, 1e-3, 1e-6)
lasso_cv = LassoCV(alphas=a_grid, random_state=1234).fit(X_poly, y)

alpha_best = lasso_cv.alpha_
mse_cv = np.mean(lasso_cv.mse_path_, axis=1)

fig, ax = plt.subplots(1,1)
ax.plot(a_grid, mse_cv)
ax.axvline(alpha_best, color='red')
ax.set_title('Lasso CV, alpha selection')









