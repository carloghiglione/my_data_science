# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:34:53 2022

@author: Utente
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('dataset/LoansNumerical.csv')
target = 'safe_loans'

y = df[target]
X = df[df.columns[df.columns != target]]


##################################################################################
##################################################################################
# Perform Logistic Regression

# imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

np.random.seed(1234)

# normalize
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns) 
    
# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True)

# Logistic Regression
C = 1e11   # high for low regularization
mod_lr = LogisticRegression(penalty='l2', C=C)
mod_lr.fit(X_train, y_train)

# predict
y_hat_train = mod_lr.predict(X_train)
y_hat_test = mod_lr.predict(X_test)

# evaluate
acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)

# cross_validation
cv = KFold(n_splits=10, shuffle=True)
mod_cv = LogisticRegression(penalty='l2', C=C)
acc_list = cross_val_score(mod_cv, X_train, y_train, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_cv_std = np.std(acc_list)

# visualize results
print('---------------')
print('Accuracy')
print(f'Train: {acc_train}')
print(f'Test: {acc_test}')
print(f'CV: mean={acc_cv}, std={acc_cv_std}')

# confusion matrix
n_test = len(y_test)
cm = confusion_matrix(y_test, y_hat_test)/n_test
fig, ax = plt.subplots(1,1)
sns.heatmap(cm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Test Set')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')

# predict with different probability
def predict_alpha(alpha, probs):
    return [1 if p >= alpha else -1 for p in probs]

y_hat_p_test = mod_lr.predict_proba(X_test)
alpha = 0.7
y_hat_alpha_test = predict_alpha(alpha, y_hat_p_test[:,1])

# Precision-Recall Curve (PR-AUC)
y_hat_p_test = mod_lr.predict_proba(X_test)
prec, rec, thresh = precision_recall_curve(y_test, y_hat_p_test[:,1])
pr_auc = average_precision_score(y_test, y_hat_p_test[:,1])

fig, ax = plt.subplots(1,1)
ax.plot(rec, prec, label='Curve')
ax.plot(rec[:-1], thresh, label='Thresh')
ax.set_ylim((-0.05, 1.05))
ax.set_xlim((-0.05, 1.05))
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()

# ROC Curve (ROC-AUC)
roc_auc = roc_auc_score(y_test, y_hat_p_test[:,1])
fpr, tpr, thresh = roc_curve(y_test, y_hat_p_test[:,1])

fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='Curve')
ax.plot([0.0,1.0],[0.0,1.0], label='Baseline')
ax.set_xlabel('False Positve Rate')
ax.set_ylabel('True Positve Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.set_title('ROC Curve')


# feature importance
coef_df = pd.DataFrame({'vars':X.columns.values, 'coef': mod_lr.coef_[0]})
coef_df['coef_abs'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('coef_abs', ascending=False)
top = 10
fig, ax = plt.subplots(1,1)
sns.barplot(x=coef_df['vars'][:top], y=coef_df['coef'][:top], color='royalblue', ax=ax)
ax.set_title('Coefficients')
ax.set_xticklabels(coef_df['vars'][:top], rotation=90)




 







    

