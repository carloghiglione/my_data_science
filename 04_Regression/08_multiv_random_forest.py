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
from sklearn.model_selection import cross_val_score, KFold, train_test_split

plt.style.use('seaborn')


###################################################################################
# Read data
df = pd.read_csv('data/house_price_train.csv', index_col=0)

y = df['SalePrice']
X = df.loc[:,'MSSubClass':'YrSold']
# X = df.loc[:,'MSSubClass':'SaleCondition_Partial']

n = X.shape[0]
r = X.shape[1]

# normalize
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) 
    y = pd.Series(StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1234)

r = X_train.shape[1]
n_train = X_train.shape[0]
n_test = X_test.shape[0]
    

    
#####################################################################################
# Random Forest Regression
n_trees = 50
max_depth = None
mod_rf = RandomForestRegressor(n_estimators=n_trees, max_depth=None, random_state=1234)
mod_rf.fit(X_train, y_train)

y_hat_train = mod_rf.predict(X_train)
y_hat_test = mod_rf.predict(X_test)

r2_train = r2_score(y_train, y_hat_train)
rmse_train = np.sqrt(np.mean( (y_train - y_hat_train)**2 ))

r2_test = r2_score(y_test, y_hat_test)
rmse_test = np.sqrt(np.mean( (y_test - y_hat_test)**2 ))


#######################################
# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_rf_cv = RandomForestRegressor(n_estimators=n_trees, max_depth=None, random_state=1234)

# R2
r2_list = cross_val_score(mod_rf_cv, X_train, y_train, cv=cv, scoring='r2')
r2_cv = np.mean(r2_list)
r2_std_cv = np.std(r2_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(mod_rf_cv, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_cv = np.mean(rmse_list)
rmse_std_cv = np.std(rmse_list)


##################################
# Visualize Performance
print('\nRandom Forest Regression, R2:')
print(f'Train: {r2_train}') 
print(f'CV: {r2_cv},  std={r2_std_cv}')
print(f'Test: {r2_test}')
print('\nRandom Forest Regression, RMSE:')
print(f'Train: {rmse_train}')
print(f'CV: {rmse_cv},  std={rmse_std_cv}')
print(f'Test: {rmse_test}')


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



