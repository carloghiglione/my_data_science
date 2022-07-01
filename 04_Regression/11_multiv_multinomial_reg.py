# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:35:03 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


################################################################################
# Read Data
iris = datasets.load_iris()
target = np.array(iris.target)

X = iris.data[:,:2]
y = target

r = X.shape[1]
n = X.shape[0]

####################################################################################
# Multinomial Regression - One Versus Rest
# this means that it performs binary logistic regression for each class

C_reg = 1e11
mod_ovr = LogisticRegression(C=C_reg, multi_class='ovr', random_state=1234)
mod_ovr.fit(X, y)

y_hat = mod_ovr.predict(X)
y_hat_p = mod_ovr.predict_proba(X)

acc = accuracy_score(y, y_hat)

cm_norm = confusion_matrix(y, y_hat)/n

fig, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(cm_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Norm')
plt.tight_layout()


cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_ovr_cv = LogisticRegression(C=C_reg, multi_class='ovr', random_state=1234)

acc_list_ovr = cross_val_score(mod_ovr_cv, X, y, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list_ovr)
acc_cv_std = np.std(acc_list_ovr)

print(f'Accuracy: {acc}')
print(f'KF-CV Accuracy: mean={acc_cv}, std={acc_cv_std}')


##############
# Plot decision boundaries
x0_r = np.max(X[:,0]) - np.min(X[:,0])
x1_r = np.max(X[:,1]) - np.min(X[:,1])
a = 0.25
h = 0.01
xx0 = np.arange(np.min(X[:,0])-a*x0_r, np.max(X[:,0])+a*x0_r, h)
xx1 = np.arange(np.min(X[:,1])-a*x1_r, np.max(X[:,1])+a*x1_r, h)

xx0_g, xx1_g = np.meshgrid(xx0, xx1)

z = mod_ovr.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z = z.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx0, xx1, z, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax)
ax.set_title('Fitted Class Space, One Versus Rest')



##################################################################################
# Multinomial Regression - Multinomial
# this means that it fits a model with softmax

C_reg = 1e11
mod_mul = LogisticRegression(C=C_reg, multi_class='multinomial', random_state=1234)
mod_mul.fit(X, y)

y_hat = mod_mul.predict(X)
y_hat_p = mod_mul.predict_proba(X)

acc = accuracy_score(y, y_hat)

c_norm = confusion_matrix(y, y_hat)/n

fig, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(c_norm, annot=True, square=True, cmap='Blues', ax = ax)
ax.set_title('Confusion Matrix Norm')
fig.tight_layout()


cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_mul_cv = LogisticRegression(C=C_reg, multi_class='multinomial', random_state=1234)


acc_list_mul = cross_val_score(mod_mul_cv, X, y, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list_mul)
acc_cv_std = np.std(acc_list_mul)

print(f'Accuracy: {acc}')
print(f'KF-CV Accuracy: mean={acc_cv}, std={acc_cv_std}')


##############
# Plot decision boundaries
x0_r = np.max(X[:,0]) - np.min(X[:,0])
x1_r = np.max(X[:,1]) - np.min(X[:,1])
a = 0.25
h = 0.01
xx0 = np.arange(np.min(X[:,0])-a*x0_r, np.max(X[:,0])+a*x0_r, h)
xx1 = np.arange(np.min(X[:,1])-a*x1_r, np.max(X[:,1])+a*x1_r, h)

xx0_g, xx1_g = np.meshgrid(xx0, xx1)

z = mod_mul.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z = z.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx0, xx1, z, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax)
ax.set_title('Fitted Class Space, Multinomial')


#################################################################################
# Compare the two approaches

test_par = ttest_rel(acc_list_ovr, acc_list_mul)
test_nonpar = wilcoxon(acc_list_ovr, acc_list_mul)

print('\nModel Comparison')
print(f'T-Test: pvalue={test_par[1]}')
print(f'Wicoxon Test: pvalue={test_nonpar[1]}')




