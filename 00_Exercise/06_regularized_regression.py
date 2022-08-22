# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:40:35 2022

@author: Utente
"""

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/house_price.csv', index_col=0)
y = df['SalePrice']
X = df.loc[:,'MSSubClass':'YrSold']

######################
# Goal: perform regularized regression with Ridge & Lasso

# imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1234)

# normalize 
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    y = StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,)
    
# split training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True)


# find Ridge or Lasso param with cross-validation
grid = np.logspace(-4, 2, 500)
is_ridge = True

if is_ridge:
    reg_cv = RidgeCV(alphas=grid).fit(X_train, y_train)
else:
    reg_cv = LassoCV(alphas=grid).fit(X_train, y_train)
alpha_reg = reg_cv.alpha_

# fit Ridge/Lasso Model
if is_ridge:
    mod_reg = Ridge(alpha=alpha_reg)
else:
    mod_reg = Lasso(alpha=alpha_reg)
mod_reg.fit(X_train, y_train)

# prediction
y_hat_train = mod_reg.predict(X_train)
y_hat_test = mod_reg.predict(X_test)

# evaluate
r2_train = r2_score(y_train, y_hat_train)
r2_test = r2_score(y_test, y_hat_test)
rmse_train = np.sqrt(np.mean( (y_hat_train - y_train)**2 ))
rmse_test = np.sqrt(np.mean( (y_hat_test - y_test)**2 ))

# cross_validation
if is_ridge:
    mod_reg_cv = Ridge(alpha=alpha_reg)
else:
    mod_reg_cv = Lasso(alpha=alpha_reg)
cv = KFold(n_splits=10, shuffle=True)
r2_list = cross_val_score(mod_reg_cv, X_train, y_train, scoring='r2')
r2_cv = np.mean(r2_list)
r2_cv_std = np.std(r2_list)
rmse_list = np.sqrt( - cross_val_score(mod_reg_cv, X_train, y_train, scoring='neg_mean_squared_error') )
rmse_cv = np.mean(rmse_list)
rmse_cv_std = np.std(rmse_list)

# visualize evaluation
print('-------------')
print('R2')
print(f'Train: {r2_train}')
print(f'cv: mean={r2_cv}, std={r2_cv_std}')
print(f'Test: {r2_test}')
print('-------------')
print('RMSE')
print(f'Train: {rmse_train}')
print(f'cv: mean={rmse_cv}, std={rmse_cv_std}')
print(f'Test: {rmse_test}')

# feature importance
df_coef = pd.DataFrame({'coef':mod_reg.coef_})
df_coef['vars'] = X.columns
df_coef['coef_abs'] = df_coef['coef'].abs()
df_coef = df_coef.sort_values('coef_abs', ascending=False)
top_vars = 10

fig, ax = plt.subplots(1,1)
sns.barplot(x=df_coef['vars'][:top_vars], y=df_coef['coef'][:top_vars], color='royalblue', ax=ax)
ax.set_title('Feature Coefficients')
ax.set_xticklabels(df_coef['vars'][:top_vars], rotation=90)










