# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:01:39 2022

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
from patsy import dmatrix, build_design_matrices

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
# Natural Cubic Spline
r = 7

data = {"x": x}
X = dmatrix("cr(x, df=r)", data, return_type='matrix')

mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X, y)

y_hat = mod.predict(X)

r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
rss = sum( (y_hat-y)**2 )
mse = np.mean( (y - y_hat)**2 )


# Plot
x_range = np.max(x)-np.min(x)
a = 0.05
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
new_data = {"x": xx}
XX = build_design_matrices([X.design_info], new_data)[0]
yy = mod.predict(XX)
conf = 0.95
alp = 1 - conf
tq = stats.t(df=(n-r-1)).ppf(1-alp/2)
IC_len = tq * np.sqrt( mse * ( 1 + np.diag( XX @ np.linalg.inv(XX.T @ XX) @ XX.T) ) )


fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(xx, yy, color='red')
ax.fill_between(xx.reshape(-1,), yy - IC_len, yy + IC_len, color='red', alpha=0.2)
ax.set_title('Fitted Model')



#######################################################################################
# Linear Regression (Statsmodels)
#X = sm.add_constant(X)
mod_sm = sm.OLS(y, X).fit()

XX_IC = mod_sm.get_prediction(XX).conf_int(0.95)

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(xx, mod_sm.predict(XX), color='red')
ax.fill_between(xx.reshape(-1,), XX_IC[:,0], XX_IC[:,1], color='red', alpha=0.2)
ax.set_title('Fitted Model')


print(mod_sm.summary())
print(f'R2-adj: {mod_sm.rsquared_adj}')

#########################################################################################
# Model Diagnostic 

# Shapiro Test on Residuals
print(f'Shapiro Test on residuals: p-value={shapiro(mod_sm.resid)[1]}')

# Diagnostic Plot
fig, ax = plt.subplots(2,2)
fig.suptitle("Model Diagnostic")

ax[0,0].scatter(mod_sm.fittedvalues, mod_sm.resid_pearson, marker='o')
ax[0,0].axhline(3, color='red', ls='--')
ax[0,0].axhline(-3, color='red', ls='--')
ax[0,0].set_title('Std Residuals vs Fitted')

sm.qqplot(mod_sm.resid, line='45', fit=True, ax=ax[0,1])
ax[0,1].set_title('QQPlot of Residuals')

sns.distplot(mod_sm.resid_pearson, bins=15, kde=True, ax=ax[1,0], label='Std Resid')
xx = np.arange(-4, 4, 0.01)
ax[1,0].plot(xx, norm.pdf(xx, 0, 1), label='N(0,1)')
ax[1,0].set_title('Std Residuals Histogram')
ax[1,0].legend()

ax[1,1].scatter(mod_sm.get_influence().hat_matrix_diag, mod_sm.resid_pearson)
ax[1,1].axhline(y=0, color='grey', linestyle='dashed')
ax[1,1].set_xlabel('Leverage')
ax[1,1].set_ylabel('Std residuals')
ax[1,1].set_title('Residuals vs Leverage Plot')

fig.tight_layout()

  
# Cook's Distance Influence plot
fig, ax = plt.subplots(1,1)
sm.graphics.influence_plot(mod_sm, criterion="cooks", ax=ax)
fig.tight_layout()
