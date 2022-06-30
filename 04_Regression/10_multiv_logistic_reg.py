# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 21:01:23 2022

@author: Utente
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay

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


##################################################################################
# Logistic Regression
C_reg = 1e11   # small values for high reguarization, set very high for no regularization
mod_lr = LogisticRegression(penalty='l2', C=C_reg)
mod_lr.fit(X, y)

y_hat = mod_lr.predict(X)
y_hat_prob = mod_lr.predict_proba(X)

acc = accuracy_score(y, y_hat)
prec = precision_score(y, y_hat)  # % correctly classified as positive wrt all that are classified as positive TP/(TP+FP)
rec = recall_score(y, y_hat)      # % correctly classified as positive wrt all that are trulery positive  TP/(TP+FN)
f1 = f1_score(y, y_hat) 
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'Recall: {rec}')
print(f'F1: {f1}')


cm = confusion_matrix(y, y_hat)
fig, ax = plt.subplots(1,1, figsize=(4,3))
sns.heatmap(cm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix')
plt.tight_layout()

cm_norm = 100*confusion_matrix(y, y_hat)/n
fig, ax = plt.subplots(1,1, figsize=(4,3))
sns.heatmap(cm_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix %')
plt.tight_layout()



####################################################################################
# Cross-Validation on Accuracy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_lr_cv = LogisticRegression(penalty='l2', C=C_reg)

acc_list = cross_val_score(mod_lr_cv, X, y, cv=cv, scoring='accuracy') # 'accuracy', 'f1', 'precision', 'recall'
acc_cv = np.mean(acc_list)
acc_cv_std = np.std(acc_list)
print(f'KF-CV: Accuracy = {acc_cv}, std={acc_cv_std}')



####################################################################################
# Cross Validation Parameter C

# from sklearn.model_selection import GridSearchCV

# parameters = {'C': 10**np.arange(1,12, dtype=np.int64)}
# log_reg_model = LogisticRegression(penalty='l2')
# cv_lr = GridSearchCV(log_reg_model, parameters, scoring='accuracy')
# cv_lr.fit(X, y)
# cv_lr.best_params_


###################################################################################
# Logistic Regression with Probability Threshold
C_reg = 1e11   # small values for high reguarization
mod_lr = LogisticRegression(penalty='l2', C=C_reg)
mod_lr.fit(X, y)

y_hat_prob = mod_lr.predict_proba(X)

def LogRegThreshold(probs, thresh):
    return [1 if p > thresh else -1 for p in probs]

thresh = 0.75
y_hat = LogRegThreshold(y_hat_prob[:,1], thresh)


acc = accuracy_score(y, y_hat)
prec = precision_score(y, y_hat)  # % correctly classified
rec = recall_score(y, y_hat)      # % 
f1 = f1_score(y, y_hat) 
print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'Recall: {rec}')
print(f'F1: {f1}')


cm_norm = 100*confusion_matrix(y, y_hat)/n
fig, ax = plt.subplots(1,1, figsize=(4,3))
sns.heatmap(cm_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title(f'Confusion Matrix %, thresh={thresh}')
plt.tight_layout()



####################################################################################
# Precision-Recall Curve (AUC)
y_true = y
y_hat = mod_lr.predict(X)
y_hat_p = mod_lr.predict_proba(X)
prec, rec, thresh = precision_recall_curve(y_true=y, probas_pred=y_hat_p[:,1])
auc = average_precision_score(y, y_hat_p[:,1])

print('AUC: {auc}')
fig, ax = plt.subplots(1,1)
ax.plot(rec, prec, label='Prec-Rec')
ax.plot(rec[:-1], thresh, label='Thresh')
ax.legend()
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall, AUC={auc}')



####################################################################################
# True-Positive Rate & False-Positive Rate (ROC) 
fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_hat_p[:,1])
roc_auc = roc_auc_score(y_true=y, y_score=y_hat_p[:,1])

print(f'ROC-AUC: {roc_auc}')
fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='ROC')
ax.plot([0.0,1.0],[0.0,1.0],label='Baseline')
ax.legend()
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title(f'ROC Curve, AUC={roc_auc}')





