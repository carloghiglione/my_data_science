# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:18:42 2022

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

plt.style.use('seaborn')


###################################################################################
# Read data
df = pd.read_csv('data/house_price_train.csv', index_col=0)

y = df['SalePrice']
# X = df.loc[:,'MSSubClass':'YrSold']
X = df.loc[:,'MSSubClass':'SaleCondition_Partial']

n = X.shape[0]
r = X.shape[1]

# normalize
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) 
    y = pd.Series(StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,))
    

    
#####################################################################################
# Random Forest Regression
n_trees = 50
mod_rf = RandomForestRegressor(n_estimators=n_trees, random_state=1234)
mod_rf.fit(X, y)

y_hat = mod_rf.predict(X)

r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
rss = sum( (y_hat-y)**2 )
mse = np.mean( (y - y_hat)**2 )

print(f'R2-adj: {r2}')
print(f'RMSE: {np.sqrt(mse)}')


#####################################################################################
# Cross-Validation

cv = KFold(n_splits=10, shuffle=True, random_state=1)
rf_reg = RandomForestRegressor(n_estimators=n_trees, random_state=1234)

# R2
r2_list = cross_val_score(rf_reg, X, y, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n - 1)/(n - r - 1)

kf_r2 = np.mean(r2_list)
kf_r2_adj = np.mean(r2_adj_list)
kf_r2_adj_std = np.std(r2_adj_list)

print(f'K-Fold CV: R2-adj = {kf_r2}, std={kf_r2_adj_std}')

# RMSE
rmse_list = np.sqrt( - cross_val_score(rf_reg, X, y, cv=cv, scoring='neg_mean_squared_error'))
kf_rmse = np.mean(rmse_list)
kf_rmse_std = np.std(rmse_list)

print(f'K-Fold CV: RMSE = {kf_rmse}, std={kf_rmse_std}')


####################################################################################
# Variable Inspection
f_imp = pd.Series(mod_rf.feature_importances_, index=X.columns) 

imp_thresh = 1e-2

f_top = f_imp[f_imp > imp_thresh].sort_values(ascending=False)

fig, ax = plt.subplots(1,1)
sns.barplot(x=f_top.index, y=f_top.values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('Feature Importance')
fig.tight_layout()










