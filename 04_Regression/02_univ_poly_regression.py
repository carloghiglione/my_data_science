# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:35:27 2022

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
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings('ignore')

####################################################################################
# Read the data
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# df = df.drop(index=[374, 401, 414])
x = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV']

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.set_title('Data')

n = len(x)


#####################################################################################
# Polynomial Regression (Sklearn)
poly_ord = 3
polynomial = PolynomialFeatures(degree=poly_ord, include_bias=False)
X_poly = polynomial.fit_transform(x)

mod_poly =  linear_model.LinearRegression()
mod_poly.fit(X_poly, y)

y_hat = mod_poly.predict(X_poly)

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
xx = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)
yy = mod_poly.predict(polynomial.fit_transform(xx))
ax.plot(xx, yy, color='red')
ax.set_title(f'Fitted Model Poly, order={poly_ord}')

r2_poly = r2_score(y, y_hat)
r2_adj_poly = 1 - (1 - r2_poly)*(n - 1)/(n - poly_ord - 1)
rss_poly = sum( (y_hat-y)**2 )


#####################################################################################
# Polynomial Regression (Statsmodels)
poly_ord = 3
polynomial = PolynomialFeatures(degree=poly_ord, include_bias=False)
X_poly = polynomial.fit_transform(x)

X = sm.add_constant(X_poly)
mod_sm = sm.OLS(y, X).fit()

x_range = np.max(x)-np.min(x)
a = 0.10
xx = np.arange(np.min(x)-a*x_range, np.max(x)+a*x_range, step=0.1).reshape(-1,1)
XX_poly = sm.add_constant(polynomial.fit_transform(xx))
XX_IC = mod_sm.get_prediction(XX_poly).conf_int(0.95)

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(xx, mod_sm.predict(XX_poly), color='red')
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



#####################################################################################
# Polynomial order selection
r2_adj_list = []
rmse_list = []

k_max = 11
for k in range(1,k_max):
    polynomial = PolynomialFeatures(degree=k, include_bias=False)
    X_k = polynomial.fit_transform(x)
    X_k = sm.add_constant(X_k)
    mod_k = sm.OLS(y, X_k).fit()
    r2_adj_list.append(mod_k.rsquared_adj)
    # rss_list.append(mod_k.ssr)
    rmse_list.append(np.sqrt(mod_k.mse_model))

fig, ax = plt.subplots(1,2)
ax[0].plot(np.arange(1,k_max), r2_adj_list)
ax[0].set_title('R2-Adjusted')
ax[1].plot(np.arange(1,k_max), rmse_list)
ax[1].set_title('Root MSE')
fig.tight_layout()




