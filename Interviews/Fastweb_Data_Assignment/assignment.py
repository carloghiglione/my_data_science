# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:51:16 2022

@author: Utente
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# dataset directory
data_dir = os.path.join(os.getcwd(), 'data', 'bank-dataset.csv')

# load data
df = pd.read_csv(data_dir)


#####################################################################################
# Step 1: Dataset Preprocessing

# size of the dataset
N = df.shape[0]
d = df.shape[1]-1

# numerical attributes
col_num = df.select_dtypes(include=[np.number]).columns

# categorical attributes
col_cat = df.select_dtypes(exclude=[np.number]).columns

# I check for wrong entries
# for col in col_cat:
#     print('-------')
#     print(f'{col}')
#     print(df[col].value_counts())

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
col_cat = df.select_dtypes(exclude=[np.number]).columns
col_cat = col_cat[col_cat != 'y']

# rewrite pdays
new_pdays = np.zeros(N)
new_pdays[df['pdays'] != -1] = 1/(df['pdays'][df['pdays'] != -1])
df['pdays'] = new_pdays

# one hot encoding of categorical variables
yy_yes_no = df['y']
yy = np.array([1 if x == 'yes' else 0 for x in yy_yes_no])


# logaritmic transform
# for col in col_num:
#     df[col] = np.log(1 - np.min(df[col].values) + df[col].values)



XX = pd.get_dummies(df.drop(columns=['y']))




################################################################################
# Step 2: Preliminary Analysis

# distribution
yy_distr = yy_yes_no.value_counts()/N

fig, ax = plt.subplots(1,1, figsize=(6,4))
sns.barplot(x=yy_distr.index, y=yy_distr.values, color='royalblue', ax=ax)
ax.set_ylim((-0.01, 1.01))
ax.set_title('Target distribution')
fig.tight_layout()

print('Target distribution')
print(yy_distr)


# display the plots for each categorical variable
# for col in col_cat:
    
#     var = df[[col, 'y']]
#     var_distr = var.groupby(col).value_counts(normalize=True).reset_index()
#     var_distr.columns = [col, 'y', 'proportion']
    
#     fig, ax = plt.subplots(1,1, figsize=(6,4))
#     sns.barplot(data=var_distr, x=col, y='proportion', hue='y')
#     ax.set_title(f'{col} Distribution')
#     plt.xticks(rotation=90)
#     fig.tight_layout()
#     plt.plot()


# display the plots for each numerical variable
# for col in col_num:

#     fig, ax = plt.subplots(1,1, figsize=(6,4))
#     sns.boxplot(data=df, x='y', y=df[col]+1)
#     ax.set_title(f'{col} Distribution')
#     ax.set_yscale('log')
#     plt.xticks(rotation=90)
#     fig.tight_layout()
#     plt.plot()



################################################################################
# Step 3: Model Development

# train test split
# set seed for reproducibility
np.random.seed(1234)

# train test split with stratification to preserve the distribution of y in both datasets
X_train, X_test, y_train, y_test = train_test_split(XX, yy, train_size=0.80, shuffle=True, stratify=yy)

cv = KFold(n_splits=5)

##################################################################
# Model 1: Logistic Regression

mod1_cv = LogisticRegressionCV(cv=5, penalty='l2', class_weight='balanced', scoring='f1')
mod1_cv.fit(X_train, y_train)

best_C = mod1_cv.C_[0]
mod1 = LogisticRegression(C=best_C, penalty='l2', class_weight='balanced')
mod1.fit(X_train, y_train)

y_hat_train = mod1.predict(X_train)

ris_mod1_train = {
    'Precision': round(precision_score(y_train, y_hat_train), 4),
    'Recall': round(recall_score(y_train, y_hat_train), 4),
    'F1-Score': round(f1_score(y_train, y_hat_train), 4),
    'Accuracy': round(accuracy_score(y_train, y_hat_train), 4)
    }

mod1_cv = LogisticRegression(C=best_C, penalty='l2', class_weight='balanced')
cv_ris_mod1 = cross_validate(mod1_cv, X_train, y_train, cv=cv, scoring=['precision', 'recall', 'f1', 'accuracy'])

ris_mod1_cv = {
    'Precision': round(np.mean(cv_ris_mod1['test_precision']), 4),
    'Recall': round(np.mean(cv_ris_mod1['test_recall']), 4),
    'F1-Score': round(np.mean(cv_ris_mod1['test_f1']), 4),
    'Accuracy': round(np.mean(cv_ris_mod1['test_accuracy']), 4)
    }


print('------------------------------')
print('Weighted Logistic Regression')
print('----------')
print('Training set results')
print(ris_mod1_train)
print('----------')
print('Cross-validation results')
print(ris_mod1_cv)
print('\n')


################################################################################
# Model 2: Random Forest

# param_grid = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth': [None, 3, 5, 10, 15, 20, 30], 'class_weight':['balanced']}
# mod_cv = RandomForestClassifier()
# gs_cv = GridSearchCV(mod_cv, param_grid, cv=5, scoring='f1')
# gs_cv.fit(X_train, y_train)

# best_depth = gs_cv.best_params_['max_depth']
# best_n_trees = gs_cv.best_params_['n_estimators']

best_max_depth = 15
best_n_trees = 200

mod2 = RandomForestClassifier(n_estimators=best_n_trees, max_depth=best_max_depth, class_weight='balanced')
mod2.fit(X_train, y_train)

y_hat_train = mod2.predict(X_train)


ris_mod2_train = {
    'Precision': round(precision_score(y_train, y_hat_train), 4),
    'Recall': round(recall_score(y_train, y_hat_train), 4),
    'F1-Score': round(f1_score(y_train, y_hat_train), 4),
    'Accuracy': round(accuracy_score(y_train, y_hat_train), 4)
    }

mod2_cv = RandomForestClassifier(n_estimators=best_n_trees, max_depth=best_max_depth, class_weight='balanced')
cv_ris_mod2 = cross_validate(mod2_cv, X_train, y_train, cv=cv, scoring=['precision', 'recall', 'f1', 'accuracy'])

ris_mod2_cv = {
    'Precision': round(np.mean(cv_ris_mod2['test_precision']), 4),
    'Recall': round(np.mean(cv_ris_mod2['test_recall']), 4),
    'F1-Score': round(np.mean(cv_ris_mod2['test_f1']), 4),
    'Accuracy': round(np.mean(cv_ris_mod2['test_accuracy']), 4)
    }


print('------------------------------')
print('Model 2: Weighted Random Forest Classifier')
print('----------')
print('Training set results')
print(ris_mod2_train)
print('----------')
print('Cross-validation results')
print(ris_mod2_cv)
print('\n')


################################################################################
# Model 3: Gradient Boosting Classifier

balanced_weights = len(y_train) / (2 * np.bincount(y_train))
sample_weight = [balanced_weights[0] if y == 0 else balanced_weights[1] for y in y_train]

# param_grid = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth': [None, 3, 5, 10, 15, 20, 30]}
# mod_cv = GradientBoostingClassifier()
# gs_cv = GridSearchCV(mod_cv, param_grid, cv=5, scoring='f1', verbose=3)
# gs_cv.fit(X_train, y_train, sample_weight=sample_weight)

# best_max_depth = gs_cv.best_params_['max_depth']
# best_n_trees = gs_cv.best_params_['n_estimators']

best_max_depth = 10
best_n_trees = 100

mod3 = GradientBoostingClassifier(n_estimators=best_n_trees, max_depth=best_max_depth)

mod3.fit(X_train, y_train, sample_weight=sample_weight)



y_hat_train = mod3.predict(X_train)


ris_mod3_train = {
    'Precision': round(precision_score(y_train, y_hat_train), 4),
    'Recall': round(recall_score(y_train, y_hat_train), 4),
    'F1-Score': round(f1_score(y_train, y_hat_train), 4),
    'Accuracy': round(accuracy_score(y_train, y_hat_train), 4)
    }

mod3_cv = GradientBoostingClassifier(n_estimators=best_n_trees, max_depth=best_max_depth)
cv_ris_mod3 = cross_validate(mod3_cv, X_train, y_train, cv=cv, scoring=['precision', 'recall', 'f1', 'accuracy'], fit_params={'sample_weight': sample_weight})

ris_mod3_cv = {
    'Precision': round(np.mean(cv_ris_mod3['test_precision']), 4),
    'Recall': round(np.mean(cv_ris_mod3['test_recall']), 4),
    'F1-Score': round(np.mean(cv_ris_mod3['test_f1']), 4),
    'Accuracy': round(np.mean(cv_ris_mod3['test_accuracy']), 4)
    }


print('------------------------------')
print('Model 3: Weighted Gradient Boosting')
print('----------')
print('Training set results')
print(ris_mod3_train)
print('----------')
print('Cross-validation results')
print(ris_mod3_cv)
print('\n')


##############################################################################
# Model selection

mod_list = [mod1, mod2, mod3]

ris_test_list = []

for i in range(3):
    y_hat_test = mod_list[i].predict(X_test)

    ris_test = {
        'Precision': round(precision_score(y_test, y_hat_test), 4),
        'Recall': round(recall_score(y_test, y_hat_test), 4),
        'F1-Score': round(f1_score(y_test, y_hat_test), 4),
        'Accuracy': round(accuracy_score(y_test, y_hat_test), 4)
        }
    ris_test_list.append(ris_test)
    

perf_recap_cv = pd.DataFrame(ris_test_list, index=['mod 1', 'mod 2', 'mod 3'])

print(perf_recap_cv)

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score


pr_auc_list = []
roc_auc_list = []
fig, ax = plt.subplots(1,2, figsize=(12,5))
for i in range(3):

    y_hat = mod_list[i].predict(X_test)
    y_hat_p = mod_list[i].predict_proba(X_test)
    
    
    prec, rec, thresh = precision_recall_curve(y_true=y_test, probas_pred=y_hat_p[:,1])
    pr_auc = average_precision_score(y_test, y_hat_p[:,1])
    pr_auc_list.append(pr_auc)
    
    fpr, tpr, thresh = roc_curve(y_true=y_test, y_score=y_hat_p[:,1])
    roc_auc = roc_auc_score(y_test, y_hat_p[:,1])
    roc_auc_list.append(roc_auc)
    
    ax[0].plot(rec, prec, label=f'Mod {i+1}, AUC: {round(pr_auc, 4)}')
    ax[1].plot(fpr, tpr, label=f'Mod {i+1}, AUC: {round(roc_auc, 4)}')

ax[0].legend()
ax[0].set_xlabel('Recall')
ax[0].set_ylabel('Precision')
ax[0].set_title('Precision-Recall Curve')

ax[1].plot([0.0,1.0],[0.0,1.0], label='Baseline')
ax[1].legend()
ax[1].set_xlabel('FPR')
ax[1].set_ylabel('TPR')
ax[1].set_title('ROC Curve')

fig.tight_layout()

df_auc = pd.DataFrame({'PR-AUC': pr_auc_list, 'ROC-AUC': roc_auc_list}, index=['mod 1', 'mod 2', 'mod 3'])

print(df_auc)



###################################################################
# Test
mod_final = mod2

y_hat_test= mod_final.predict(X_test)


print('------------------------------')
print('Final Model')
print('----------')
print('Test set results')
print(ris_test)


cm = confusion_matrix(y_test, y_hat_test)/len(y_test)

fig, ax = plt.subplots(1,1)
sns.heatmap(cm, cmap='Blues', annot=True, square=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - Test Set')



# feature importance
feat_imp = pd.Series(mod_final.feature_importances_, index=XX.columns)
feat_imp = feat_imp.sort_values(ascending=False)

fig, ax = plt.subplots(1,1)
sns.barplot(x=feat_imp[:8].index, y=feat_imp[:8].values, ax=ax, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('Final Model - Feature Importance')