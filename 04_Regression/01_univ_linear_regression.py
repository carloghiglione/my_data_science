# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:39:59 2022

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

####################################################################################
# Read the data
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
x = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV']

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.set_title('Data')

n = len(x)
r = 1


#####################################################################################
# Linear Regression (Sklearn)
mod = linear_model.LinearRegression()
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
ax.plot(xx, yy, color='red')
ax.fill_between(xx.reshape(-1,), yy - IC_len, yy + IC_len, color='red', alpha=0.2)
ax.set_title('Fitted Model')


#######################################################################################
# Linear Regression (Statsmodels)
X = sm.add_constant(x)
mod_sm = sm.OLS(y, X).fit()

x_range = np.max(x)-np.min(x)
a = 0.10
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
XX = sm.add_constant(xx)
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



####################################################################################
# Model Evaluation on a Test Set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
mod_train = linear_model.LinearRegression()
mod_train.fit(X_train, y_train)

y_hat_test = mod_train.predict(X_test)
r2_test = r2_score(y_test, y_hat_test)
rmse_test = np.sqrt(np.mean( (y_test - y_hat_test)**2 ))



####################################################################################
# Model Evaluation with K-Fold Crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

cv = cv=KFold(n_splits=10, shuffle=True, random_state=1234)
lin_reg = linear_model.LinearRegression()

# R2
r2_list = cross_val_score(lin_reg, x, y, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n - 1)/(n - r - 1)

kf_r2 = np.mean(r2_list)
kf_r2_adj = np.mean(r2_adj_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(lin_reg, X, y, cv=cv, scoring='neg_mean_squared_error'))
kf_rmse = np.mean(rmse_list)


print(f'K-Fold CV: R2-adj = {kf_r2_adj}')
print(f'K-Fold CV: RMSE = {kf_rmse}')


kf = KFold(n_splits=10, shuffle=True, random_state=1234)
rmse_list = []
for split_train, split_test in kf.split(X,y):
    X_train = sm.add_constant(X[split_train])
    y_train = y[split_train]
    X_test = sm.add_constant(X[split_test])
    y_test = y[split_test]
    mod_kf = sm.OLS(y_train, X_train).fit()
    rmse = np.sqrt(np.mean( ( y_test - mod_kf.predict(X_test) )**2 ))
    rmse_list.append(rmse)



