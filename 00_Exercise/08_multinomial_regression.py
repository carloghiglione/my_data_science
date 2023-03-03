# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:01:01 2022

@author: Utente
"""

from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

iris = datasets.load_iris()
target = np.array(iris.target)

X = pd.DataFrame(iris.data[:,:2], columns=['X0','X1']) 
y = target

###################################################################################
###################################################################################
# Perform multinomial regression

np.random.seed(1234)

# imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# normalize
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True)


# Multinomial Regression
C = 1e11
mod = LogisticRegression(penalty='l2', C=C, multi_class='multinomial')  # 'ovr' also possible
mod.fit(X_train, y_train)

y_hat_train = mod.predict(X_train)
y_hat_test = mod.predict(X_test)

acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)

# cross-validation
cv = KFold(n_splits=10, shuffle=True)
mod_cv = LogisticRegression(penalty='l2', C=C, multi_class='multinomial')
acc_list = cross_val_score(mod_cv, X_train, y_train, cv=cv, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_cv_std = np.std(acc_list)

# visualize performance
print('-------------')
print('Accuracy')
print(f'Train: {acc_train}')
print(f'Test: {acc_test}')
print(f'CV: mean={acc_cv}, std={acc_cv_std}')

# confusion matrix
n_test = len(y_test)
cm = confusion_matrix(y_test, y_hat_test)/n_test
fig, ax = plt.subplots(1,1)
sns.heatmap(cm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Test')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')




######################
# visualize prediction region
XX = X.values
x0_r = np.max(XX[:,0]) - np.min(XX[:,0])
x1_r = np.max(XX[:,1]) - np.min(XX[:,1])
a = 0.25
h = 0.1
xx0 = np.arange(np.min(XX[:,0])-a*x0_r, np.max(XX[:,0])+a*x0_r, h)
xx1 = np.arange(np.min(XX[:,1])-a*x1_r, np.max(XX[:,1])+a*x1_r, h)
xx0_g, xx1_g = np.meshgrid(xx0, xx1)

z = mod.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z = z.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx0, xx1, z, cmap='tab10')
sns.scatterplot(x=X['X0'], y=X['X1'], hue=y, palette='tab10', ax=ax)
ax.set_title('Fitted Class')



















