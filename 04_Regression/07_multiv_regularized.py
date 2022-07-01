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
from sklearn.model_selection import cross_val_score, KFold, train_test_split

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


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



####################################################################################
# Ridge Regression

# CV to select best alpha
a_grid = np.arange(1, 50, 1e-1)
ridge_cv = RidgeCV(alphas=a_grid, store_cv_values=True).fit(X_train, y_train)
alpha_best_l2 = ridge_cv.alpha_
mse_cv = np.mean(ridge_cv.cv_values_, axis=0)

fig, ax = plt.subplots(1,1)
ax.plot(a_grid, mse_cv)
ax.axvline(alpha_best_l2, color='red')
ax.set_title('Ridge CV, alpha selection')


# fit model with optimal alpha
mod_ridge = Ridge(alpha=alpha_best_l2, random_state=1234)
mod_ridge.fit(X_train, y_train)

y_hat_train = mod_ridge.predict(X_train)
y_hat_test = mod_ridge.predict(X_test)

r2_train = r2_score(y_train, y_hat_train)
r2_adj_train = 1 - (1 - r2_train)*(n_train - 1)/(n_train - r - 1)
rmse_train = np.sqrt(np.mean( (y_train - y_hat_train)**2 ))

r2_test = r2_score(y_test, y_hat_test)
r2_adj_test = 1 - (1 - r2_test)*(n_test - 1)/(n_test - r - 1)
rmse_test = np.sqrt(np.mean( (y_test - y_hat_test)**2 ))


#######################################
# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_ridge_cv = Ridge(alpha=alpha_best_l2, random_state=1234)

# R2
r2_list = cross_val_score(mod_ridge_cv, X_train, y_train, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n_train - 1)/(n_train - r - 1)
r2_cv = np.mean(r2_list)
r2_std_cv = np.std(r2_list)
r2_adj_cv = np.mean(r2_adj_list)
r2_adj_std_cv = np.std(r2_adj_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(mod_ridge_cv, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_cv = np.mean(rmse_list)
rmse_std_cv = np.std(rmse_list)


##################################
# Visualize Performance
print('\nRidge Regression, R2:')
print(f'Train: {r2_train}') #, R2-adj={r2_adj_train}')
print(f'CV: {r2_cv},  std={r2_std_cv}')
#print(f'CV: R2-adj={r2_adj_cv},  std={r2_adj_std_cv}')
print(f'Test: {r2_test}') #,  R2-adj={r2_adj_test}')
print('\nRidge Regression, RMSE:')
print(f'Train: {rmse_train}')
print(f'CV: {rmse_cv},  std={rmse_std_cv}')
print(f'Test: {rmse_test}')


###################################
# Diagnostic Plot
res = y_train - y_hat_train
res_std = (res - np.mean(res))/np.std(res, ddof=(r+1))

fig, ax = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("Model Diagnostic Ridge")

ax[0].scatter(y_hat_train, res_std, marker='o')
ax[0].axhline(3, color='red', ls='--')
ax[0].axhline(-3, color='red', ls='--')
ax[0].set_title('Std Residuals vs Fitted')

sm.qqplot(res_std, line='45', fit=True, ax=ax[1])
ax[1].set_title('QQPlot of Residuals')

sns.distplot(res_std, bins=15, kde=True, ax=ax[2], label='Std Resid')
xx = np.arange(-4, 4, 0.01)
ax[2].plot(xx, norm.pdf(xx, 0, 1), label='N(0,1)')
ax[2].set_title('Std Residuals Histogram')
ax[2].legend()
fig.tight_layout()


###################################################################################
# Feature Selection with Ridge
coef = pd.Series(mod_ridge.coef_, index = X.columns)
coef_top = coef[coef.abs() > 1e-4]

print(f'Selected variables: {coef_top.count()} out of {coef.count()}')

coef_abs = coef_top.abs().sort_values(ascending=False)

df1 = pd.DataFrame({'coef': coef_top.values}, index=coef_top.index)
df2 = pd.DataFrame({'coef_abs': coef_abs.values}, index=coef_abs.index)
df_coef= pd.merge(df1, df2, left_index=True, right_index=True)
df_coef = df_coef.sort_values(by='coef_abs', ascending=False)

top_plot = 20
fig, ax = plt.subplots(1,1)
sns.barplot(x=df_coef.index[:top_plot], y=df_coef['coef'][:top_plot], ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('Most Influential Coeffs, Ridge')
fig.tight_layout()



# ####################################################################################
# Lasso Regression

# CV to select best alpha
a_grid = np.arange(1e-3, 1, 1e-3)
lasso_cv = LassoCV(alphas=a_grid).fit(X_train, y_train)
alpha_best_l1 = lasso_cv.alpha_


# fit model with optimal alpha
mod_lasso = Lasso(alpha=alpha_best_l1, random_state=1234)
mod_lasso.fit(X_train, y_train)

y_hat_train = mod_lasso.predict(X_train)
y_hat_test = mod_lasso.predict(X_test)

r2_train = r2_score(y_train, y_hat_train)
r2_adj_train = 1 - (1 - r2_train)*(n_train - 1)/(n_train - r - 1)
rmse_train = np.sqrt(np.mean( (y_train - y_hat_train)**2 ))

r2_test = r2_score(y_test, y_hat_test)
r2_adj_test = 1 - (1 - r2_test)*(n_test - 1)/(n_test - r - 1)
rmse_test = np.sqrt(np.mean( (y_test - y_hat_test)**2 ))


#######################################
# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_lasso_cv = Lasso(alpha=alpha_best_l1, random_state=1234)

# R2
r2_list = cross_val_score(mod_lasso_cv, X_train, y_train, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n_train - 1)/(n_train - r - 1)
r2_cv = np.mean(r2_list)
r2_std_cv = np.std(r2_list)
r2_adj_cv = np.mean(r2_adj_list)
r2_adj_std_cv = np.std(r2_adj_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(mod_ridge_cv, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_cv = np.mean(rmse_list)
rmse_std_cv = np.std(rmse_list)


##################################
# Visualize Performance
print('\nLasso Regression, R2:')
print(f'Train: {r2_train}') #, R2-adj={r2_adj_train}')
print(f'CV: {r2_cv},  std={r2_std_cv}')
#print(f'CV: R2-adj={r2_adj_cv},  std={r2_adj_std_cv}')
print(f'Test: {r2_test}') #,  R2-adj={r2_adj_test}')
print('\nLasso Regression, RMSE:')
print(f'Train: {rmse_train}')
print(f'CV: {rmse_cv},  std={rmse_std_cv}')
print(f'Test: {rmse_test}')


###################################
# Diagnostic Plot
res = y_train - y_hat_train
res_std = (res - np.mean(res))/np.std(res, ddof=(r+1))

fig, ax = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("Model Diagnostic Lasso")

ax[0].scatter(y_hat_train, res_std, marker='o')
ax[0].axhline(3, color='red', ls='--')
ax[0].axhline(-3, color='red', ls='--')
ax[0].set_title('Std Residuals vs Fitted')

sm.qqplot(res_std, line='45', fit=True, ax=ax[1])
ax[1].set_title('QQPlot of Residuals')

sns.distplot(res_std, bins=15, kde=True, ax=ax[2], label='Std Resid')
xx = np.arange(-4, 4, 0.01)
ax[2].plot(xx, norm.pdf(xx, 0, 1), label='N(0,1)')
ax[2].set_title('Std Residuals Histogram')
ax[2].legend()
fig.tight_layout()


###################################################################################
# Feature Selection with Lasso
coef = pd.Series(mod_lasso.coef_, index = X.columns)
coef_top = coef[coef.abs() > 1e-4]

print(f'Selected variables: {coef_top.count()} out of {coef.count()}')

coef_abs = coef_top.abs().sort_values(ascending=False)

df1 = pd.DataFrame({'coef': coef_top.values}, index=coef_top.index)
df2 = pd.DataFrame({'coef_abs': coef_abs.values}, index=coef_abs.index)
df_coef= pd.merge(df1, df2, left_index=True, right_index=True)
df_coef = df_coef.sort_values(by='coef_abs', ascending=False)

top_plot = 20
fig, ax = plt.subplots(1,1)
sns.barplot(x=df_coef.index[:top_plot], y=df_coef['coef'][:top_plot], ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('Most Influential Coeffs, Lasso')
fig.tight_layout()



#################################################################################
# Compare Ridge and Lasso
cv = KFold(n_splits=10, shuffle=True, random_state=1)
lin_lasso = Lasso(alpha=alpha_best_l1, random_state=1234)
lin_ridge = Ridge(alpha=alpha_best_l2, random_state=1234)

rmse_list_l1 = np.sqrt( - cross_val_score(lin_lasso, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_list_l2 = np.sqrt( - cross_val_score(lin_ridge, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))

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




