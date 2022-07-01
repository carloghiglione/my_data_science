# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:46:38 2022

@author: Utente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


#################################################################################
# Read Data
df = pd.read_csv('data/LoansNumerical.csv')
target = 'safe_loans'
y = df[target]
X = df[df.columns[df.columns != target]]

n = X.shape[0]
r = X.shape[1]

normalization = True
if normalization:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1234)

r = X_train.shape[1]
n = X_train.shape[0]


##################################################################################
# Ada Boost 
n_estim = 50
max_depth = 3
crit = 'gini'  # 'entropy'
mod_ab = AdaBoostClassifier(DecisionTreeClassifier(criterion=crit, max_depth=max_depth), n_estimators=n_estim, learning_rate=1.0)
mod_ab.fit(X_train, y_train)

y_hat_train = mod_ab.predict(X_train)
y_hat_test = mod_ab.predict(X_test)

acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)


# Cross-Validation (very long)
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_ab_cv = AdaBoostClassifier(DecisionTreeClassifier(criterion=crit, max_depth=max_depth), n_estimators=n_estim, learning_rate=1.0)

acc_list = cross_val_score(mod_ab_cv, X_train, y_train, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_std_cv = np.std(acc_list)

print('Accuracy Ada Boost')
print(f'Train: {acc_train}')
print(f'Train CV: mean={acc_cv},  std={acc_std_cv}')
print(f'Test: {acc_test}')


# Confusion Matrix
cm_norm = confusion_matrix(y_test, y_hat_test)/len(y_test)

fig, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(cm_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Norm, Random Forest (Test)')
plt.tight_layout()


# Feature Importance
feat_imp = pd.Series(mod_ab.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(1,1)
sns.barplot(x=feat_imp.index, y=feat_imp.values, ax=ax)
plt.xticks(rotation=90)
ax.set_title('Feature Importance')
fig.tight_layout()







