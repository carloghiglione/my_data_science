# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:26:06 2022

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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1234)


# divide training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, shuffle=True)


# fit the model
n_classes = np.unique(y).shape[0]
prior = np.ones(n_classes)*(1/n_classes)

# bayes
mod = GaussianNB(priors=prior)
mod_no_prior = GaussianNB() # priors are empirical frequencies

# LDA
mod = LinearDiscriminantAnalysis(priors=prior)
mod_no_prior = LinearDiscriminantAnalysis()

# QDA
mod = QuadraticDiscriminantAnalysis(priors=prior)
mod_no_prior = QuadraticDiscriminantAnalysis()

mod.fit(X_train, y_train)
mod_no_prior.fit(X_train, y_train)


# predict
y_hat_train = mod.predict(X_train)
y_hat_test = mod.predict(X_test)

acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)

# cross-validation
mod_cv = GaussianNB(priors=prior)
cv = KFold(n_splits=10, shuffle=True)
acc_list = cross_val_score(mod_cv, X_train, y_train, scoring='accuracy')
acc_cv = np.mean(acc_list)
acc_cv_std = np.std(acc_list)

# visualize results
print('Accuracy')
print(f'Train: {acc_train}')
print(f'Test: {acc_test}')
print(f'CV: mean={acc_cv}, std={acc_cv_std}')

# confusion matrix
n_test = y_test.shape[0]
cm = confusion_matrix(y_test, y_hat_test)/n_test

fig, ax = plt.subplots(1,1)
sns.heatmap(cm, cmap='Blues', annot=True, square=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Classification Test set')











