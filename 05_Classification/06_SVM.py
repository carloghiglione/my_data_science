# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:18:54 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel
import pandas as pd

plt.style.use('seaborn')
import warnings 
warnings.filterwarnings("ignore")


################################################################################
# Read Data
iris = datasets.load_iris()
target = np.array(iris.target)

X = iris.data[:,:2]
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1234)

r = X_train.shape[1]
n = X_train.shape[0]


################################################################################
# SVM
C_reg = 1e11
mod_svm = SVC(C=C_reg, kernel='rbf')  # kernel{''linear', 'poly' (should specify order), 'rbf', 'sigmoid', 'precomputed'} 
mod_svm.fit(X_train, y_train)

y_hat_train = mod_svm.predict(X_train)
y_hat_test = mod_svm.predict(X_test)

acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)


# CrossValidation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_svm_cv = SVC(C=C_reg, kernel='rbf') 

acc_list = cross_val_score(mod_svm_cv, X_train, y_train, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_std_cv = np.std(acc_list)

print('Accuracy LDA')
print(f'Train: {acc_train}')
print(f'Train CV: mean={acc_cv},  std={acc_std_cv}')
print(f'Test: {acc_test}')


# Confusion Matrix
cmat_norm = confusion_matrix(y_test, y_hat_test)/len(y_test)

fig, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(cmat_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Norm, SVM (Test)')
plt.tight_layout()


##############
# Plot decision boundaries
x0_r = np.max(X[:,0]) - np.min(X[:,0])
x1_r = np.max(X[:,1]) - np.min(X[:,1])
a = 0.25
h = 0.01
xx0 = np.arange(np.min(X[:,0])-a*x0_r, np.max(X[:,0])+a*x0_r, h)
xx1 = np.arange(np.min(X[:,1])-a*x1_r, np.max(X[:,1])+a*x1_r, h)

xx0_g, xx1_g = np.meshgrid(xx0, xx1)

z = mod_svm.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z = z.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx0, xx1, z, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax)
ax.set_title('Fitted Class Space SVM')
fig.tight_layout()














