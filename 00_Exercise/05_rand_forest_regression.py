# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:01:11 2022

@author: Utente
"""

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/house_price.csv', index_col=0)
y = df['SalePrice']
X = df.loc[:,'MSSubClass':'YrSold']

######################
# Goal: perform regression with random forest

# imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# normalize data
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    y = StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,)

# split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True, random_state=1234)


# Random Forest Regression
n_trees = 50
max_depth = None
mod_rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=1234)
mod_rf.fit(X_train, y_train)

# predict
y_hat_train = mod_rf.predict(X_train)
y_hat_test = mod_rf.predict(X_test)

# evaluate performance on train and test
r2_train = r2_score(y_train, y_hat_train)
r2_test = r2_score(y_test, y_hat_test)
rmse_train = np.sqrt(np.mean( (y_hat_train - y_train)**2 ))
rmse_test = np.sqrt(np.mean( (y_hat_test - y_test)**2 ))

# cross-validation
mod_rf_cv = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, )
cv = KFold(n_splits=10, shuffle=True, random_state=1234)
r2_list = cross_val_score(mod_rf_cv, X_train, y_train, cv=cv, scoring='r2')
r2_cv = np.mean(r2_list)
r2_cv_std = np.std(r2_list)
rmse_list = np.sqrt( - cross_val_score(mod_rf_cv, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_cv = np.mean(rmse_list)
rmse_cv_std = np.std(rmse_list)

# visualize perfromance
print('-------------------')
print('r2')
print(f'train: {r2_train}')
print(f'cv: {r2_cv}, std: {r2_cv_std}')
print(f'Test: {r2_test}')
print('-------------------')
print('rmse')
print(f'train: {rmse_train}')
print(f'cv: {rmse_cv}, std: {rmse_cv_std}')
print(f'Test: {rmse_test}')

# feature importance
feat_imp = pd.Series(mod_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_top = 10

fig, ax = plt.subplots(1,1)
sns.barplot(x=feat_imp.index[:feat_top], y=feat_imp.values[:feat_top], ax=ax, color='royalblue')
ax.set_title('Feature Importance')
ax.set_xticklabels(feat_imp.index[:feat_top], rotation=90)






    