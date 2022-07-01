# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:14:06 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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


##################################################################################
# LDA
# Requires each group to be Multivariate Gaussian, all groups with same Covariance Matrix
priors = np.ones(3)/3
mod_lda = LinearDiscriminantAnalysis(priors=priors)  # can introduce shrinkage with shrinkage='auto' 
mod_lda_noprior = LinearDiscriminantAnalysis()       # priors are empirical frequencies


mod_lda.fit(X_train, y_train)

#posteriors = mod_lda.class_count_ / n


y_hat_train = mod_lda.predict(X_train)
y_hat_test = mod_lda.predict(X_test)
acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)

posteriors = pd.DataFrame({'class': y_hat_train}).groupby('class').size()/len(y_train)

# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_lda_cv = LinearDiscriminantAnalysis(priors=priors)

acc_list = cross_val_score(mod_lda_cv, X_train, y_train, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_std_cv = np.std(acc_list)

print('Accuracy')
print(f'Train: {acc_train}')
print(f'Train CV: mean={acc_cv},  std={acc_std_cv}')
print(f'Test: {acc_test}')

print(f'Posteriors: {posteriors}')


# Confusion Matrix
cm_norm = confusion_matrix(y_test, y_hat_test)/len(y_test)

fig, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(cm_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Norm, Testl')
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

z = mod_lda.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z = z.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx0, xx1, z, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax)
ax.set_title(f'Fitted Class Space LDA')
fig.tight_layout()



