# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:33:46 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1234)

r = X_train.shape[1]
n = X_train.shape[0]


#################################################################################
# KNN Classification (Uniform Weights)
k = 10
mod_knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='uniform') # different algos 'auto', 'ball_tree', 'kd_tree' (faster), 'brute'
mod_knn.fit(X_train, y_train)

y_hat_train = mod_knn.predict(X_train)
y_hat_test = mod_knn.predict(X_test)

acc_train = accuracy_score(y_train, y_hat_train)
acc_test = accuracy_score(y_test, y_hat_test)

# Cross-Validation
cv = KFold(n_splits=10, shuffle=True, random_state=1)
mod_knn_cv = KNeighborsClassifier(n_neighbors=k, algorithm='auto')

acc_list = cross_val_score(mod_knn_cv, X_train, y_train, cv=cv)
acc_train_cv = np.mean(acc_list)
acc_train_std_cv = np.std(acc_list)

print(f'KNN, Uniform Weight, k={k}')
print(f'Accuracy Train: {acc_train}')
print(f'Accuracy Train CV: mean={acc_train_cv},  std={acc_train_std_cv}')
print(f'Accuracy Test: {acc_test}')


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

z = mod_knn.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z = z.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,1)
ax.pcolormesh(xx0, xx1, z, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax)
ax.set_title(f'Fitted Class Space KNN, k = {k}')
fig.tight_layout()



######################################################################################
# Optimal K Selection
k_min = 5
k_max = 30

acc_unif = []
acc_dist = []
cv = KFold(n_splits=10, shuffle=True, random_state=1)

for k in range(k_min, k_max+1):
    mod_k_unif = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='uniform')
    mod_k_dist = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='distance')
    
    acc_unif_list = cross_val_score(mod_k_unif, X_train, y_train, cv=cv, scoring='accuracy')
    acc_dist_list = cross_val_score(mod_k_dist, X_train, y_train, cv=cv, scoring='accuracy')
    
    acc_unif.append(np.mean(acc_unif_list))
    acc_dist.append(np.mean(acc_dist_list))

k_best_unif = k_min + np.argmax(acc_unif)
k_best_dist = k_min + np.argmax(acc_dist)
print(f'Best k Unif: {k_best_unif}')
print(f'Best k Dist: {k_best_dist}')

fig, ax = plt.subplots(1,1)
ax.plot(np.arange(k_min, k_max+1), acc_unif, marker='o', ls='-', label='Unif')
ax.plot(np.arange(k_min, k_max+1), acc_dist, marker='o', ls='-', label='Dist')
ax.legend()
ax.set_title('K Selection, Accuracy')
fig.tight_layout()



#####################################################################################
# Model Comparison
mod_knn_unif = KNeighborsClassifier(n_neighbors=k_best_unif, algorithm='auto', weights='uniform')
mod_knn_unif.fit(X_train, y_train)

mod_knn_dist = KNeighborsClassifier(n_neighbors=k_best_dist, algorithm='auto', weights='distance')
mod_knn_dist.fit(X_train, y_train)

y_hat_test_unif = mod_knn_unif.predict(X_test)
y_hat_test_dist = mod_knn_dist.predict(X_test)

acc_test_unif = accuracy_score(y_test, y_hat_test_unif)
acc_test_dist = accuracy_score(y_test, y_hat_test_dist)

print('Test Accuracy')
print(f'Uniform: {acc_test_unif}')
print(f'Distance: {acc_test_dist}')


##############
# Plot decision boundaries
x0_r = np.max(X[:,0]) - np.min(X[:,0])
x1_r = np.max(X[:,1]) - np.min(X[:,1])
a = 0.25
h = 0.01
xx0 = np.arange(np.min(X[:,0])-a*x0_r, np.max(X[:,0])+a*x0_r, h)
xx1 = np.arange(np.min(X[:,1])-a*x1_r, np.max(X[:,1])+a*x1_r, h)

xx0_g, xx1_g = np.meshgrid(xx0, xx1)

z_unif = mod_knn_unif.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z_unif = z_unif.reshape(xx0_g.shape)
z_dist = mod_knn_dist.predict(np.c_[xx0_g.ravel(), xx1_g.ravel()])
z_dist = z_dist.reshape(xx0_g.shape)

fig, ax = plt.subplots(1,2, figsize=(12,6))

ax[0].pcolormesh(xx0, xx1, z_unif, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax[0])
ax[0].set_title('Fitted Class Space KNN Unif')

ax[1].pcolormesh(xx0, xx1, z_dist, cmap='tab10')
sns.scatterplot(X[:,0], X[:,1], hue=y, palette='tab10', ax=ax[1])
ax[1].set_title('Fitted Class Space KNN Dist')

fig.tight_layout()






