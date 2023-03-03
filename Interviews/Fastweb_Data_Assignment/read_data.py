# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:54:17 2022

@author: Utente
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import PolynomialFeatures

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# dataset directory
data_dir = os.path.join(os.getcwd(), 'data', 'bank-dataset.csv')

# load data
df = pd.read_csv(data_dir)


#####################################################################################
# Dataset Preprocessing

# size of the dataset
N = df.shape[0]
d = df.shape[1]-1

# numerical attributes
col_num = df.select_dtypes(include=[np.number]).columns

# categorical attributes
col_cat = df.select_dtypes(exclude=[np.number]).columns


for col in col_cat:
    print('-------')
    print(f'{col}')
    print(df[col].value_counts())

# I see 'marital' variable has some wrong entries (whose real value is evident), I correct them
df['marital'][df['marital']=='Singl'] = 'single'
df['marital'][df['marital']=='divrcd'] = 'divorced'
df['marital'][df['marital']=='S'] = 'single'
df['marital'][df['marital']=='Single'] = 'single'


# percentage of NANs for numerical attributes
num_nan = df[col_num].isnull().sum()/N
num_nan = num_nan.sort_values(ascending=False)

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=num_nan.index, y=num_nan.values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of Nans in numerical variables')
fig.tight_layout()

# fill nans with the mean value
for col in col_num:
    df[col] = df[col].fillna(df[col].mean())


# percentage of NANs for categorical attributes
cat_nan = df[col_cat].isnull().sum()/N
cat_nan = num_nan.sort_values(ascending=False)

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=cat_nan.index, y=cat_nan.values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of Nans in categorical variables')
fig.tight_layout()

# fill nans with the median value
for col in col_cat:
    df[col] = df[col].fillna(df[col].mode()[0])


# percentage of "unknown" in categorical variables
cat_ukn = df[col_cat].apply(lambda x: np.sum(x == 'unknown'))/N
cat_ukn = cat_ukn.sort_values(ascending=False)

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=cat_ukn.index, y=cat_ukn.values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of Unknown in categorical variables')
fig.tight_layout()


# I create a new attribute expressing the season of the last contact

def season(df_row):
    day = df_row['day']
    month = df_row['month']
    if (month in ['jan', 'feb']) or (month == 'dec' and day >= 21) or (month == 'mar' and day < 21):
        return 'winter'
    elif (month in ['apr', 'may']) or (month == 'mar' and day >= 21) or (month == 'jun' and day < 21):
        return 'spring'
    elif (month in ['jul', 'aug']) or (month == 'jun' and day >= 21) or (month == 'sep' and day < 23):
        return 'summer'
    else:
        return 'autumn'

df['season'] = df[['day', 'month']].apply(season, axis=1)


new_pdays = np.zeros(N)
new_pdays[df['pdays'] != -1] = 1/(df['pdays'][df['pdays'] != -1])
df['pdays'] = new_pdays

# I remove the following attributes:
#   -"poutcome" for its too high percentage of "unknown" (larger than 80%)
#   -"day" beacause the number of the day of the month is not a meaningful variable by itself
#   I could have exploited it to find the day of the week but I do not know the year when the dataset refers to
#   I could have exploited it also to compute the distance of the last contact from the birthday of the users, but the birthday is not available too
#   -"month" in order to avoid high dimensionality due to 12 value categories sintetized in "season"

#df = df.drop(columns=['poutcome', 'day'])
# df = df.drop(columns=['day'])

# in the remaining attributes, 'unknown' value is an additional label of the categorical variable


#########################################################################################
# Output Analysis
yy = df['y']
XX = df.drop(columns=['y'])

yy_distr = yy.value_counts()/N

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=yy_distr.index, y=yy_distr.values, color='royalblue', ax=ax)
ax.set_ylim((-0.01, 1.01))
ax.set_title('Target distribution')
fig.tight_layout()



#########################################################################################
# One hot encoding
yy = np.array([1 if x == 'yes' else 0 for x in yy])
XX = pd.get_dummies(XX)


##########################################################################################
# train test split

# set seed for reproducibility
np.random.seed(1234)

# train test split with stratification to preserve the distribution of y in both datasets
X_train, X_test, y_train, y_test = train_test_split(XX, yy, train_size=0.80, shuffle=True, stratify=yy)



##########################################################################################
# Model 1
# Logistic Regression

mod1_cv = LogisticRegressionCV(cv=5, penalty='l2', class_weight=None, scoring='f1')
mod1_cv.fit(X_train, y_train)

best_C = mod1_cv.C_
mod1 = LogisticRegression(C=best_C, penalty='l2', class_weight='balanced')


y_hat_train = mod1.predict(X_train)
y_hat_test = mod1.predict(X_test)

prec_train = precision_score(y_train, y_hat_train)
rec_train = recall_score(y_train, y_hat_train)
f1_train = f1_score(y_train, y_hat_train)

prec_test = precision_score(y_test, y_hat_test)
rec_test = recall_score(y_test, y_hat_test)
f1_test = f1_score(y_test, y_hat_test)

print('------------------------------')
print('Logistic Regression')
print('----------')
print('Training set results')
print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
print('----------')
print('Test set results')
print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')
print('\n')

# pd.Series(y_hat_train).value_counts()



##########################################################################################
# Model 2
# Weighted Logistic Regression

mod = LogisticRegressionCV(cv=5, penalty='l2', class_weight='balanced', scoring='f1')
mod.fit(X_train, y_train)

y_hat_train = mod.predict(X_train)
y_hat_test = mod.predict(X_test)

prec_train = precision_score(y_train, y_hat_train)
rec_train = recall_score(y_train, y_hat_train)
f1_train = f1_score(y_train, y_hat_train)

prec_test = precision_score(y_test, y_hat_test)
rec_test = recall_score(y_test, y_hat_test)
f1_test = f1_score(y_test, y_hat_test)

print('------------------------------')
print('Weighted Logistic Regression')
print('----------')
print('Training set results')
print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
print('----------')
print('Test set results')
print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')
print('\n')


###################################################################################
# Model 3
# Random Forest   # opt: balanced, n_trees=200, max_depth=15
n_trees = 200
max_depth = 15
mod = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, class_weight='balanced')
mod.fit(X_train, y_train)

# y_hat_train = mod.predict(X_train)
# y_hat_test = mod.predict(X_test)


y_hat_train_prob = mod.predict_proba(X_train)[:,1]
y_hat_test_prob = mod.predict_proba(X_test)[:,1]

p = 0.5
y_hat_train = [1 if prob > p else 0 for prob in y_hat_train_prob]
y_hat_test= [1 if prob > p else 0 for prob in y_hat_test_prob]


prec_train = precision_score(y_train, y_hat_train)
rec_train = recall_score(y_train, y_hat_train)
f1_train = f1_score(y_train, y_hat_train)

prec_test = precision_score(y_test, y_hat_test)
rec_test = recall_score(y_test, y_hat_test)
f1_test = f1_score(y_test, y_hat_test)

print('------------------------------')
print(f'Weighted Random Forest | n_trees={n_trees} | max_depth={max_depth}')
print('----------')
print('Training set results')
print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
print('----------')
print('Test set results')
print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')


#####################################################################################
# grid search
# param_grid = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth': [None, 3, 5, 10, 15, 20], 'class_weight':['balanced']}
# mod_cv = RandomForestClassifier()
# gs_cv = GridSearchCV(mod_cv, param_grid, cv=5, scoring='f1', verbose=3)
# gs_cv.fit(X_train, y_train)

# gs_cv.best_params_



# feature importnace
feat_imp = pd.Series(mod.feature_importances_, index=XX.columns)
feat_imp = feat_imp.sort_values(ascending=False)

fig, ax = plt.subplots(1,1)
sns.barplot(x=feat_imp[:5].index, y=feat_imp[:5].values, ax=ax)
plt.xticks(rotation=90)






















# #################################################################################

# from sklearn.naive_bayes import CategoricalNB

# col_cat = df.select_dtypes(exclude=[np.number]).columns

# df_2 = df.drop(columns=['y'])
# col_cat = df_2.select_dtypes(exclude=[np.number]).columns
# col_num = df_2.select_dtypes(include=[np.number]).columns
# df_2[col_cat] = df_2[col_cat].apply(lambda x: x.astype('category').cat.codes)

# np.random.seed(1234)
# X2_train, X2_test, y2_train, y2_test = train_test_split(df_2, yy, train_size=0.8, shuffle=True)

# mod_nb_cat = CategoricalNB()

# mod_nb_cat.fit(X2_train[col_cat], y_train)


# y_hat_train_c = mod_nb_cat.predict(X2_train[col_cat])
# y_hat_test_c = mod_nb_cat.predict(X2_test[col_cat])

# prec_train = precision_score(y_train, y_hat_train_c)
# rec_train = recall_score(y_train, y_hat_train_c)
# f1_train = f1_score(y_train, y_hat_train_c)

# prec_test = precision_score(y_test, y_hat_test_c)
# rec_test = recall_score(y_test, y_hat_test_c)
# f1_test = f1_score(y_test, y_hat_test_c)

# print('------------------------------')
# print(f'Categorical Naive Bayes')
# print('----------')
# print('Training set results')
# print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
# print('----------')
# print('Test set results')
# print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')



# mod_rf_2 = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced')



# mod_rf_2.fit(X2_train[col_num], y_train)


# y_hat_train_n = mod_rf_2.predict(X2_train[col_num])
# y_hat_test_n = mod_rf_2.predict(X2_test[col_num])

# prec_train = precision_score(y_train, y_hat_train_n)
# rec_train = recall_score(y_train, y_hat_train_n)
# f1_train = f1_score(y_train, y_hat_train_n)

# prec_test = precision_score(y_test, y_hat_test_n)
# rec_test = recall_score(y_test, y_hat_test_n)
# f1_test = f1_score(y_test, y_hat_test_n)

# print('------------------------------')
# print(f'Random forest 2')
# print('----------')
# print('Training set results')
# print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
# print('----------')
# print('Test set results')
# print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')



# ##################
# # Ensemble

# from sklearn.ensemble import VotingClassifier


# ####################
# from sklearn.ensemble import GradientBoostingClassifier

# n_trees = 100
# lr = 0.1
# max_depth = 10
# mod_gb = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=lr, max_depth=max_depth)

# balanced_weights = len(y_train) / (2 * np.bincount(y_train))
# sample_weight = [balanced_weights[0] if y == 0 else balanced_weights[1] for y in y_train]
# mod_gb.fit(X_train, y_train, sample_weight=sample_weight)

# y_hat_train = mod_gb.predict(X_train)
# y_hat_test = mod_gb.predict(X_test)

# prec_train = precision_score(y_train, y_hat_train)
# rec_train = recall_score(y_train, y_hat_train)
# f1_train = f1_score(y_train, y_hat_train)

# prec_test = precision_score(y_test, y_hat_test)
# rec_test = recall_score(y_test, y_hat_test)
# f1_test = f1_score(y_test, y_hat_test)

# print('------------------------------')
# print(f'Gradient boost | n_trees={n_trees} | lr={lr} | max_depth={max_depth}')
# print('----------')
# print('Training set results')
# print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
# print('----------')
# print('Test set results')
# print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')



##########################
# Ensemble
# from sklearn.ensemble import VotingClassifier

# mod_rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced')
# mod_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)

# eclf = VotingClassifier(estimators=[('rf', mod_rf), ('gb', mod_gb)],
#                         voting='soft', weights=balanced_weights)

# eclf.fit(X_train, y_train)


# y_hat_train = eclf.predict(X_train)
# y_hat_test = eclf.predict(X_test)

# prec_train = precision_score(y_train, y_hat_train)
# rec_train = recall_score(y_train, y_hat_train)
# f1_train = f1_score(y_train, y_hat_train)

# prec_test = precision_score(y_test, y_hat_test)
# rec_test = recall_score(y_test, y_hat_test)
# f1_test = f1_score(y_test, y_hat_test)

# print('------------------------------')
# print(f'Ensemble')
# print('----------')
# print('Training set results')
# print(f'Precision: {round(prec_train, 4)} | Recall: {round(rec_train, 4)} | F1: {round(f1_train, 4)}')
# print('----------')
# print('Test set results')
# print(f'Precision: {round(prec_test, 4)} | Recall: {round(rec_test, 4)} | F1: {round(f1_test, 4)}')



