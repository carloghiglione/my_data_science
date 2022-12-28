# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 09:15:47 2022

@author: Utente
"""

from sklearn import datasets
import warnings 
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

iris = datasets.load_iris()
target = np.array(iris.target)

X = pd.DataFrame(iris.data[:,:2], columns=['X1','X2'])
y = target


#################################################################################
#################################################################################
# imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1234)

# split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True)

# fit model
k = 5
mod = KNeighborsClassifier(n_neighbors=k, weights='uniform') #'distance'
mod.fit(X_train, y_train)

# predict
y_hat_train = mod.predict(X_train)
y_hat_test = mod.predict(X_test)

acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)

# cross-validation
mod_cv = KNeighborsClassifier(n_neighbors=k, weights='uniform')
cv = KFold(n_splits=10, shuffle=True)
acc_list = cross_val_score(mod_cv, X_train, y_train, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_cv_std = np.std(acc_list)

# results
print('----------------')
print('Accuracy')
print(f'Train: {acc_train}')
print(f'Test: {acc_test}')
print(f'CV: mean={acc_cv}, std={acc_cv_std}')

# confusion matrix
n_test = X_test.shape[0]
cm = confusion_matrix(y_test, y_hat_test)/n_test

fig, ax = plt.subplots(1,1)
sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, square=True)
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
ax.set_title('Classification Test')


















