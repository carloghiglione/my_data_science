# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:52:42 2022

@author: Utente
"""

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/house_price.csv', index_col=0)
y = df['SalePrice']
X = df.loc[:,'MSSubClass':'YrSold']

######################
# Goal: perform linear regression


#################################################################################
#################################################################################
# imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##################
# normalize data
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)
    y = pd.Series(StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,))
    
################
# split training and test datasets
train_p = 0.80
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_p, shuffle=True, random_state=1234)

################
# sizes
r = X_train.shape[1]
n = X_train.shape[0]


#################
# linear regression

# fit model with training data
mod = LinearRegression()
mod.fit(X_train, y_train)

# predict
y_hat_train = mod.predict(X_train)
y_hat_test = mod.predict(X_test)

# evaluate performance on train and test
r2_train = r2_score(y_train, y_hat_train)
rmse_train = np.sqrt(np.mean((y_train - y_hat_train)**2))
r2_test = r2_score(y_test, y_hat_test)
rmse_test = np.sqrt(np.mean((y_test - y_hat_test)**2))


#####################
# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1234)
mod_cv = LinearRegression()

# R2
r2_list = cross_val_score(mod_cv, X_train, y_train, cv=cv, scoring='r2')
r2_cv = np.mean(r2_list)
r2_cv_std = np.std(r2_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(mod_cv, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_cv = np.mean(rmse_list)
rmse_cv_std = np.std(rmse_list)


#####################
# visualize performance
print('--------------')
print('R2')
print(f'Train: {r2_train}')
print(f'CV: {r2_cv}, sd = {r2_cv_std}')
print(f'Test: {r2_test}')
print('--------------')
print('RMSE')
print(f'Train: {rmse_train}')
print(f'CV: {rmse_cv}, sd = {rmse_cv_std}')
print(f'Test: {rmse_test}')


######################
# Feature Importance (if normalized)
df_coef = pd.DataFrame({'coef': mod.coef_})
df_coef['vars'] = X_train.columns
df_coef['coef_abs'] = df_coef['coef'].abs()
df_coef = df_coef.sort_values('coef_abs', ascending=False)

top_plot = 10
df_coef = df_coef.iloc[:top_plot,:]
fig, ax = plt.subplots(1,1)
sns.barplot(x='vars', y='coef', data=df_coef, color='royalblue')
ax.set_xticklabels(df_coef['vars'], rotation=90)
ax.set_title('Most important features')


#######################
# Interaction and higher order
XX = X.iloc[:,0:3]
deg = 2
inter_only = False
poly = PolynomialFeatures(degree=deg, interaction_only=inter_only, include_bias=False)
XX_trasf = poly.fit_transform(XX)
XX_trasf_cols = poly.get_feature_names(XX.columns)
XX_trasf = pd.DataFrame(XX_trasf, columns=XX_trasf_cols)














