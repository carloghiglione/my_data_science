# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:36:49 2022

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
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use('seaborn')


###################################################################################
# Read data
df = pd.read_csv('data/house_price_train.csv', index_col=0)

y = df['SalePrice']
X = df.loc[:,'MSSubClass':'YrSold']
# X = df.loc[:,'MSSubClass':'SaleCondition_Partial'].values

n = X.shape[0]
r = X.shape[1]


####################################################################################
# Correlation Analysis

# Correlation among the continuous attributes (no division by class)
corrmat = X.corr()
fig, ax = plt.subplots(1,1)
sns.heatmap(corrmat, square=True, cmap='coolwarm', annot=False, ax=ax)
ax.set_title('Correlation')
plt.tight_layout()


# Vif Analysis
max_vif = np.inf
thresh_vif = 10
stop = False
while not(stop):
    r = X.shape[1]
    vif_list = [variance_inflation_factor(X.values, i) for i in range(r)]
    max_vif_col = np.argmax(vif_list)
    max_vif = vif_list[max_vif_col]
    if max_vif > thresh_vif:
        X = X.drop(columns=[X.columns[max_vif_col]])
    else:
        stop = True

r = X.shape[1]    

# Correlation among the continuous attributes (no division by class)
corrmat = X.corr()
fig, ax = plt.subplots(1,1)
sns.heatmap(corrmat, square=True, cmap='coolwarm', annot=False, ax=ax)
ax.set_title('Correlation')
plt.tight_layout()


####################################################################################
# Linear Regression (Statsmodels)
X = sm.add_constant(X)
mod_sm = sm.OLS(y, X).fit()

mod_sm_pvals = mod_sm.pvalues
pval_thresh = 0.05
pval_max = np.inf
stop = False
while not(stop) and len(X.columns)>1:
    mod_sm = sm.OLS(y, X).fit()
    pvals = mod_sm.pvalues
    max_pval_col = np.argmax(pvals)
    max_pval = pvals[max_pval_col]
    if max_pval > pval_thresh:
        X = X.drop(columns=[X.columns[max_pval_col]])
    else:
        stop = True
        
r = X.shape[1]

nas = [1298, 1061, 5, 916]
X = X.drop(index=nas)
y = y.drop(index=nas)
mod_sm = sm.OLS(y, X).fit()
print(mod_sm.summary())


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








