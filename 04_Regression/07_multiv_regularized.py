# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:54:20 2022

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
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
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
normalize = False
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) 
    y = pd.Series(StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,))

####################################################################################
# Ridge Regression

# CV to select best alpha
a_grid = np.arange(1, 50, 1e-1)
ridge_cv = RidgeCV(alphas=a_grid, store_cv_values=True).fit(X, y)
alpha_best_l2 = ridge_cv.alpha_
mse_cv = np.mean(ridge_cv.cv_values_, axis=0)

fig, ax = plt.subplots(1,1)
ax.plot(a_grid, mse_cv)
ax.axvline(alpha_best_l2, color='red')
ax.set_title('Ridge CV, alpha selection')


# fit model with optimal alpha
mod_ridge = Ridge(alpha=alpha_best_l2, random_state=1234)
mod_ridge.fit(X, y)

y_hat = mod_ridge.predict(X)
r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
mse = np.mean( (y - y_hat)**2 )
rmse = np.sqrt(mse)

# CV to estimate R2 and RMSE
cv = KFold(n_splits=10, shuffle=True, random_state=1)
lin_ridge = Ridge(alpha=alpha_best_l2, random_state=1234)

# R2
r2_list = cross_val_score(lin_ridge, X, y, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n - 1)/(n - r - 1)

kf_r2 = np.mean(r2_list)
kf_r2_adj = np.mean(r2_adj_list)
kf_r2_adj_std = np.std(r2_adj_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(lin_ridge, X, y, cv=cv, scoring='neg_mean_squared_error'))
kf_rmse = np.mean(rmse_list)
kf_rmse_std = np.std(rmse_list)

print('Ridge')
print(f'R2-adj = {r2_adj}')
print(f'K-Fold CV: R2-adj = {kf_r2}, std={kf_r2_adj_std}')
print(f'RMSE = {rmse}')
print(f'K-Fold CV: RMSE = {kf_rmse}, std={kf_rmse_std}')



####################################################################################
# Lasso Regression

# CV to select best alpha
a_grid = np.arange(1e-3, 1, 1e-3)
lasso_cv = LassoCV(alphas=a_grid).fit(X, y)
alpha_best_l1 = lasso_cv.alpha_
mse_cv = np.mean(lasso_cv.mse_path_, axis=1)

fig, ax = plt.subplots(1,1)
ax.plot(a_grid, mse_cv)
ax.axvline(alpha_best_l1, color='red')
ax.set_title('Ridge CV, alpha selection')


# fit model with optimal alpha
mod_lasso = Lasso(alpha=alpha_best_l1, random_state=1234)
mod_lasso.fit(X, y)

y_hat = mod_lasso.predict(X)
r2 = r2_score(y, y_hat)
r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
mse = np.mean( (y - y_hat)**2 )
rmse = np.sqrt(mse)

# CV to estimate R2 and RMSE
cv = KFold(n_splits=10, shuffle=True, random_state=1)
lin_lasso = Lasso(alpha=alpha_best_l1, random_state=1234)

# R2
r2_list = cross_val_score(lin_lasso, X, y, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n - 1)/(n - r - 1)

kf_r2 = np.mean(r2_list)
kf_r2_adj = np.mean(r2_adj_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(lin_lasso, X, y, cv=cv, scoring='neg_mean_squared_error'))
kf_rmse = np.mean(rmse_list)

print('\nLasso')
print(f'R2-adj = {r2_adj}')
print(f'K-Fold CV: R2-adj = {kf_r2}')
print(f'RMSE = {rmse}')
print(f'K-Fold CV: RMSE = {kf_rmse}')



###################################################################################
# Feature Selection with Lasso
coef = pd.Series(mod_lasso.coef_, index = X.columns)

coef_top = coef[coef.abs() > 1e-4]#.sort_values(ascending=False)

print(f'Selected variables: {coef_top.count()} out of {coef.count()}')

print(f'Selected Variables: {coef_top}')


coef_abs = coef_top.abs().sort_values(ascending=False)

df1 = pd.DataFrame({'coef': coef_top.values}, index=coef_top.index)
df2 = pd.DataFrame({'coef_abs': coef_abs.values}, index=coef_abs.index)
df_coef= pd.merge(df1, df2, left_index=True, right_index=True)
df_coef = df_coef.sort_values(by='coef_abs', ascending=False)

top_plot = 20
fig, ax = plt.subplots(1,1)
sns.barplot(x=df_coef.index[:top_plot], y=df_coef['coef'][:top_plot], ax=ax)
plt.xticks(rotation=90)
ax.set_title('Most Influential Coeffs')
fig.tight_layout()


#################################################################################
# Compare Ridge and Lasso
cv = KFold(n_splits=10, shuffle=True, random_state=1)
lin_lasso = Lasso(alpha=alpha_best_l1, random_state=1234)
lin_ridge = Ridge(alpha=alpha_best_l2, random_state=1234)

rmse_list_l1 = np.sqrt( - cross_val_score(lin_lasso, X, y, cv=cv, scoring='neg_mean_squared_error'))
rmse_list_l2 = np.sqrt( - cross_val_score(lin_ridge, X, y, cv=cv, scoring='neg_mean_squared_error'))

# Test (difference of paired data, they are paired since random_state of KFold is the same,
# they are computed on the same dataset)

# Non-parametric (Wilcoxon)
test = stats.wilcoxon(rmse_list_l2, rmse_list_l1)
print(f'Test Different Performance: p-value={test[1]}')

test_better = stats.wilcoxon(rmse_list_l2, rmse_list_l1, alternative='greater')
print(f'Test Ridge better than Lasso: p-value={test_better[1]}')

# Parameteric (T-Test)
test = stats.ttest_rel(rmse_list_l2, rmse_list_l1)
print(f'Test Different Performance: p-value={test[1]}')

test_better = stats.ttest_rel(rmse_list_l2, rmse_list_l1, alternative='greater')
print(f'Test Ridge better than Lasso: p-value={test_better[1]}')




