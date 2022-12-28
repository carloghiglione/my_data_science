# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 08:28:38 2022

@author: Utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv(os.path.join('data', 'train.csv'))
df_test = pd.read_csv(os.path.join('data', 'test.csv'))


################################################################################
# size of the dataset
N = df.shape[0]
d = df.shape[1]-2

# numerical variables
col_num = df.select_dtypes(include=[np.number]).columns

# categorical variables
col_cat = df.select_dtypes(exclude=[np.number]).columns

print(f'Dataset size | N: {N} | d: {d}')
print(f'Numerical variables: {col_num.values}')
print(f'Categorical variables: {col_cat.values}')


# visulize categorical variables
# for col in col_cat:
#     print('-------')
#     print(f'{col}')
#     print(df[col].value_counts())
    
    
    
# percentage of NANs for numerical variables
num_nan = df[col_num].isnull().sum()/N
num_nan = num_nan.sort_values(ascending=False)

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=num_nan.index, y=num_nan.values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of Nans in numerical variables')
fig.tight_layout()

# print('% of NANs in numerical variables')
# print(num_nan)


# percentage of NANs for categorical vaiables
cat_nan = df[col_cat].isnull().sum()/N
cat_nan = cat_nan.sort_values(ascending=False)

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=cat_nan.index, y=cat_nan.values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of Nans in categorical variables')
fig.tight_layout()

# print('% of NANs in categorical variables')
# print(cat_nan)


# since the number of nans is very low, I directly remove them
df = df.dropna()
N = df.shape[0]


#################################################################################
# feature extraction
df['install_timestamp'] = pd.to_datetime(df['install_timestamp'])
df['free_trial_timestamp'] = pd.to_datetime(df['free_trial_timestamp'])

# time waited to pass to premium in hours
df['time_to premium'] = (df['free_trial_timestamp'] - df['install_timestamp'])/np.timedelta64(1,'h')


# attribute a macro-region for each country
country_list = np.sort(df['country'].unique())

def macro_region(country):
    if country in ['DZ', 'EG', 'EH', 'LY', 'MA', 'SD', 'SS', 'TN', 'BF', 'BJ', 'CI', 'CV', 'GH', 'GM', 'GN', 'GW', 'LR', 'ML', 
                   'MR', 'NE', 'NG', 'SH', 'SL', 'SN', 'TG', 'AO', 'CD', 'ZR', 'CF', 'CG', 'CM', 'GA', 'GQ', 'ST', 'TD',
                   'BI', 'DJ', 'ER', 'ET', 'KE', 'KM', 'MG', 'MU', 'MW', 'MZ', 'RE', 'RW', 'SC', 'SO', 'TZ', 'UG', 'YT', 'ZM', 'ZW',
                   	'BW', 'LS', 'NA', 'SZ', 'ZA']:
        return 'Africa'
        
    if country in []:
        return 'Else-Africa'
    
    if country in ['GG', 'JE', 'AX', 'DK', 'EE', 'FI', 'FO', 'GB', 'IE', 'IM', 'IS', 'LT', 'LV', 'NO', 'SE', 'SJ']:
        return 'Northen-Europe'
    
    if country in ['AT', 'BE', 'CH', 'DE', 'DD', 'FR', 'FX', 'LI', 'LU', 'MC', 'NL']:
        return 'Western-Europe'
    
    if country in ['BG', 'BY', 'CZ', 'HU', 'MD', 'PL', 'RO', 'RU', 'SU', 'SK', 'UA', 'XK']:
        return 'Eastern-Europe'
    
    if country in ['AD', 'AL', 'BA', 'ES', 'GI', 'GR', 'HR', 'IT', 'ME', 'MK', 'MT', 'RS', 'PT', 'SI', 'SM', 'VA', 'YU']:
        return 'Southern-Europe'
    
    if country in ['BM', 'CA', 'GL', 'PM', 'US']:
        return 'Northen-America'
    
    if country in ['AG', 'AI', 'AN', 'AW', 'BB', 'BL', 'BS', 'CU', 'DM', 'DO', 'GD', 'GP', 'HT', 'JM', 'KN', 'KY', 'LC', 'MF', 'MQ', 'MS', 'PR', 'TC', 'TT', 'VC', 'VG', 'VI',
                   'BZ', 'CR', 'GT', 'HN', 'MX', 'NI', 'PA', 'SV', 'AR', 'BO', 'BR', 'CL', 'CO', 'EC', 'FK', 'GF', 'GY', 'PE', 'PY', 'SR', 'UY', 'VE']:
        return 'Latin-America-Caribbean'
    
    if country in ['AE', 'AM', 'AZ', 'BH', 'CY', 'GE', 'IL', 'IQ', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'NT', 'SY', 'TR', 'YE', 'YD', 'TM', 'TJ', 'KG', 'KZ', 'UZ', 'AF', 'MN', ]:
        return 'Western-Central-Asia'
    
    if country in ['BD', 'BT', 'IN', 'IR', 'LK', 'MV', 'NP', 'PK', 'BN', 'ID', 'KH', 'LA', 'MM', 'BU', 'MY', 'PH', 'SG', 'TH', 'TL', 'TP', 'VN', 'JP', 'CN', 'HK', 'KR', 'TW']:
        return 'Southern-Eastern-Asia'
    
    if country in ['AU', 'NF', 'NZ', 'FJ', 'NC', 'PG', 'SB', 'VU', 'FM', 'GU', 'KI', 'MH', 'MP', 'NR', 'PW', 'AS', 'CK', 'NU', 'PF', 'PN', 'TK', 'TO', 'TV', 'WF', 'WS', 'AQ']:
        return 'Oceania'
    
    else:
        return 'not-available'
    
df['macro-region'] = df['country'].apply(macro_region)


# attrubute else to nations with few users
low_card_countries = df['country'].value_counts()[df['country'].value_counts() < 100].index.values

def else_country(country):
    if country in low_card_countries:
        return 'else'

df['country'][df['country'].apply(lambda x: x in low_card_countries)] = 'else'
    

# macro division of devices
df['macro-device'] = df['device_type']
df['macro-device'][df['device_type'].str.contains('phone', case=False)] = 'iPhone'
df['macro-device'][df['device_type'].str.contains('pad', case=False)] = 'else'
df['macro-device'][df['device_type'].str.contains('pod', case=False)] = 'else'


# attribute a number to os version being low for old versions to high for newer versions
def cut_version(version):
    return version[:4]

df['os_version'] = df['os_version'].apply(cut_version)
df['os_version'] = df['os_version'].apply(np.float64)


# make periodicity of subscription categorical
df['product_periodicity'][df['product_periodicity']==7] = 'week'
df['product_periodicity'][df['product_periodicity']==30] = 'month'
df['product_periodicity'][df['product_periodicity']==365] = 'year'


# I see onboarding birth year has some irrealistic values (>= 2015), I set them to 2015
df['onboarding_birth_year'][df['onboarding_birth_year'].values > 2014] = 2015 



# reset numerical and categorical variables
col_cat = df.select_dtypes(exclude=[np.number]).columns
col_num = df.select_dtypes(include=[np.number]).columns



################################################################################
# display on at-the-time the plots for each categorical variable

do_plots = False

if do_plots:
    for col in col_cat[3:]:
        
        fig, ax = plt.subplots(1,1)
        sns.boxplot(data=df, x=col, y=df['net_purchases_1y'].values+1)
        ax.set_title(f'{col} Distribution')
        ax.set_yscale('log')
        plt.xticks(rotation=90)
        fig.tight_layout()
        plt.plot()
    
    
    
    # pairplots for numerical attributes
    fig, ax = plt.subplots(1,1, figsize=(30,20))
    pd.plotting.scatter_matrix(df[col_num], ax=ax)
    fig.tight_layout()




############################################################
# feature selection
# X = df[['country', 'language', 'device_type', 'attribution_network', 'product_periodicity', 'onboarding_gender', 'macro-region', 'macro-device',
#        'os_version', 'product_price_tier', 'product_free_trial_length', 'onboarding_birth_year', 'net_purchases_15d', 'time_to premium']]
X = df[['country','attribution_network', 'product_periodicity', 'onboarding_gender', 'macro-device',
       'os_version', 'onboarding_birth_year', 'net_purchases_15d', 'time_to premium']]
y = df['net_purchases_1y']

X = pd.get_dummies(X)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
XX_trasf = poly.fit_transform(X)
XX_trasf_cols = poly.get_feature_names(X.columns)
X = pd.DataFrame(XX_trasf, columns=XX_trasf_cols)

##############################################################
# cross-validation 
np.random.seed(1234)

# K-Fold cross validation object
cv = KFold(n_splits=5, shuffle=True)

###################################
# Model 1
from sklearn.ensemble import RandomForestRegressor


##################################
# grid search

do_grid_search = False

if do_grid_search:
    # grid of search
    param_grid = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth': [None, 3, 5, 10, 15, 20]}
    
    # model
    mod1_gs = RandomForestRegressor()
    
    # perform grid search with cross-validation
    gs_cv = GridSearchCV(mod1_gs, param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=3)
    gs_cv.fit(X, y)
    
    # best parameters
    best_max_depth = gs_cv.best_params_['max_depth']
    best_n_trees = gs_cv.best_params_['n_estimators']


##################
best_n_trees_1 = 100
best_max_depth_1 = 3

mod1 = RandomForestRegressor(n_estimators=best_n_trees_1, max_depth=best_max_depth_1)
mod1.fit(X, y)
y_hat = mod1.predict(X)
rmse_train = np.sqrt(mean_squared_error(y, y_hat))


mod1_cv = RandomForestRegressor(n_estimators=best_n_trees_1, max_depth=best_max_depth_1)
rmse_list = np.sqrt(-cross_val_score(mod1_cv, X, y, cv=cv, scoring='neg_mean_squared_error'))

print('------------------------------------')
print(f'Random Forest | n_trees: {best_n_trees_1} | max_depth: {best_max_depth_1}')
print(f'RMSE train: {round(rmse_train, 3)}')
print(f'RMSE CV: {round(np.mean(rmse_list), 3)}, std = {round(np.std(rmse_list), 3)}')




# #################################################################################
# # Model 2: Ridge Regression
# from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso

# grid = np.logspace(-4, 2, 500)

# reg_cv = RidgeCV(alphas=grid, cv=cv).fit(X, y)
# alpha_reg = reg_cv.alpha_

# # fit Ridge Model
# mod2 = Ridge(alpha=alpha_reg)
# mod2.fit(X, y)

# y_hat = mod2.predict(X)
# rmse_train = np.sqrt(mean_squared_error(y, y_hat))

# mod2_cv = Ridge(alpha=alpha_reg)
# rmse_list = np.sqrt(-cross_val_score(mod2_cv, X, y, cv=cv, scoring='neg_mean_squared_error'))

# print('------------------------------------')
# print(f'Ridge Regression | alpha: {round(alpha_reg,4)}')
# print(f'RMSE train: {round(rmse_train, 3)}')
# print(f'RMSE CV: {round(np.mean(rmse_list), 3)}, std = {round(np.std(rmse_list), 3)}')




# #################################################################################
# # Model 3: Lasso Regression
# from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso

# grid = np.logspace(-4, 2, 500)

# reg_cv = LassoCV(alphas=grid, cv=cv).fit(X, y)
# alpha_reg = reg_cv.alpha_

# # fit Lasso Model
# mod3 = Lasso(alpha=alpha_reg)
# mod3.fit(X, y)

# y_hat = mod3.predict(X)
# rmse_train = np.sqrt(mean_squared_error(y, y_hat))

# mod3_cv = Lasso(alpha=alpha_reg)
# rmse_list = np.sqrt(-cross_val_score(mod3_cv, X, y, cv=cv, scoring='neg_mean_squared_error'))

# print('------------------------------------')
# print(f'Lasso Regression | alpha: {round(alpha_reg,4)}')
# print(f'RMSE train: {round(rmse_train, 3)}')
# print(f'RMSE CV: {round(np.mean(rmse_list), 3)}, std = {round(np.std(rmse_list), 3)}')



# #################################################################################
# # Model 4: Gradient Boosting 
# from sklearn.ensemble import GradientBoostingRegressor

# do_grid_search = False
# if do_grid_search:
#     param_grid = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth': [None, 3, 5, 10, 15, 20], 'learning_rate': [0.001, 0.01, 0.1, 0.2]}
#     mod4_cv = GradientBoostingRegressor()
#     gs_cv = GridSearchCV(mod4_cv, param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=3)
#     gs_cv.fit(X, y)
    
#     best_max_depth_4 = gs_cv.best_params_['max_depth']
#     best_n_trees_4 = gs_cv.best_params_['n_estimators']
#     best_lr_4 = gs_cv.best_params_['learning_rate']


# ####################################
# best_n_trees_4 = 50
# best_max_depth_4 = 3
# best_lr = 0.1

# mod4 = GradientBoostingRegressor(n_estimators=best_n_trees_4, max_depth=best_max_depth_4, learning_rate=best_lr)
# mod4.fit(X, y)
# y_hat = mod4.predict(X)
# rmse_train = np.sqrt(mean_squared_error(y, y_hat))


# mod4_cv = GradientBoostingRegressor(n_estimators=best_n_trees_4, max_depth=best_max_depth_4, learning_rate=best_lr)
# rmse_list = np.sqrt(-cross_val_score(mod4_cv, X, y, cv=cv, scoring='neg_mean_squared_error'))

# print('------------------------------------')
# print(f'Gradient Boosting | n_trees: {best_n_trees_4} | max_depth: {best_max_depth_4} | lr: {best_lr}')
# print(f'RMSE train: {round(rmse_train, 3)}')
# print(f'RMSE CV: {round(np.mean(rmse_list), 3)}, std = {round(np.std(rmse_list), 3)}')





