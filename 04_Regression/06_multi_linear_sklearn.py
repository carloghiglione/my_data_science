# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:56:01 2022

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


#####################################################################################
# Linear Regression (Sklearn)
mod_lr = linear_model.LinearRegression()
mod_lr.fit(X_train, y_train)

y_hat_train = mod_lr.predict(X_train)
y_hat_test = mod_lr.predict(X_test)

r2_train = r2_score(y_train, y_hat_train)
r2_adj_train = 1 - (1 - r2_train)*(n_train - 1)/(n_train - r - 1)
rmse_train = np.sqrt(np.mean( (y_train - y_hat_train)**2 ))

r2_test = r2_score(y_test, y_hat_test)
r2_adj_test = 1 - (1 - r2_test)*(n_test - 1)/(n_test - r - 1)
rmse_test = np.sqrt(np.mean( (y_test - y_hat_test)**2 ))


############################
# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_lr_cv = linear_model.LinearRegression()

# R2
r2_list = cross_val_score(mod_lr_cv, X_train, y_train, cv=cv, scoring='r2')
r2_adj_list = 1 - (1 - r2_list)*(n_train - 1)/(n_train - r - 1)
r2_cv = np.mean(r2_list)
r2_std_cv = np.std(r2_list)
r2_adj_cv = np.mean(r2_adj_list)
r2_adj_std_cv = np.std(r2_adj_list)

# RMSE
rmse_list = np.sqrt( - cross_val_score(mod_lr_cv, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
rmse_cv = np.mean(rmse_list)
rmse_std_cv = np.std(rmse_list)


##################################
# Visualize Performance
print('\nLinear Regression, R2:')
print(f'Train: {r2_train}') #, R2-adj={r2_adj_train}')
print(f'CV: {r2_cv},  std={r2_std_cv}')
#print(f'CV: R2-adj={r2_adj_cv},  std={r2_adj_std_cv}')
print(f'Test: {r2_test}') #,  R2-adj={r2_adj_test}')
print('\nLinear Regression, RMSE:')
print(f'Train: {rmse_train}')
print(f'CV: {rmse_cv},  std={rmse_std_cv}')
print(f'Test: {rmse_test}')



###################################
# Feature Importance (if X, y are standardized)
coef = pd.Series(mod_lr.coef_, index = X.columns)
coef_top = coef[coef.abs() > 1e-4]

print(f'Selected variables: {coef_top.count()} out of {coef.count()}')

coef_abs = coef_top.abs().sort_values(ascending=False)

df1 = pd.DataFrame({'coef': coef_top.values}, index=coef_top.index)
df2 = pd.DataFrame({'coef_abs': coef_abs.values}, index=coef_abs.index)
df_coef= pd.merge(df1, df2, left_index=True, right_index=True)
df_coef = df_coef.sort_values(by='coef_abs', ascending=False)

top_plot = 10
fig, ax = plt.subplots(1,1)
sns.barplot(x=df_coef.index[:top_plot], y=df_coef['coef'][:top_plot], ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('Most Influential Coeffs')
fig.tight_layout()



###################################
# Diagnostic Plot
res = y_train - y_hat_train
res_std = (res - np.mean(res))/np.std(res, ddof=(r+1))

fig, ax = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("Model Diagnostic")

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



# ########################################################################################
# # Interaction
# polynomial = PolynomialFeatures(degree = 2, include_bias=False, interaction_only=True)
# X_int = pd.DataFrame(polynomial.fit_transform(X)) 
# X_int_cols = poly.get_feature_names(X.columns)
# X_int = pd.DataFrame(X_int, columns=X_int_cols)

# mod_int = linear_model.LinearRegression()
# mod_int.fit(X_int, y)

# y_hat = mod_int.predict(X_int)

# r2 = r2_score(y, y_hat)
# r2_adj = 1 - (1 - r2)*(n - 1)/(n - r - 1)
# rss = sum( (y_hat-y)**2 )
# mse = np.mean( (y - y_hat)**2 )

# print(f'R2-adj (int): {r2}')
# print(f'RMSE (int): {np.sqrt(mse)}')


