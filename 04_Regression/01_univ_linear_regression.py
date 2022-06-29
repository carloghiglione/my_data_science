# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:39:59 2022

@author: Utente
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler

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


#####################################################################################
# Linear Regression
mod = linear_model.LinearRegression()
mod.fit(x, y)

y_hat = mod.predict(x)

fig, ax = plt.subplots(1,1)
ax.scatter(x, y)
ax.plot(x, y_hat, color='red')
ax.set_title('Fitted Model')


#######################################################################################
# Model Diagnostic
mod_sm = sm.OLS(y, sm.add_constant(x)).fit()
print(mod_sm.summary())

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

ax[1,0].scatter(mod_sm.fittedvalues, np.sqrt(mod_sm.resid_pearson), marker='o')
ax[1,0].set_xlabel('Fitted values')
ax[1,0].set_ylabel('Sqrt(Std residuals)')
ax[1,0].set_title('Scale-Location Plot')

ax[1,1].scatter(mod_sm.get_influence().hat_matrix_diag, mod_sm.resid_pearson)
ax[1,1].axhline(y=0, color='grey', linestyle='dashed')
ax[1,1].set_xlabel('Leverage')
ax[1,1].set_ylabel('Std residuals')
ax[1,1].set_title('Residuals vs Leverage Plot')

fig.tight_layout()

  
# Plot Cook's distance plot
fig, ax = plt.subplots(1,1)
sm.graphics.influence_plot(mod_sm, criterion="cooks", ax=ax)
fig.tight_layout()















