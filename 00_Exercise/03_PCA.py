# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 09:48:38 2022

@author: Utente
"""
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/tourists.txt', sep=' ')
tab = df.iloc[:,2:10]
tab_cols = tab.columns
n_cols = len(tab_cols)
df = df.rename(columns={'Month':'class'})

######################
# Goal: perform PCA on tourists.txt dataset


#################################################################################
#################################################################################
# imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('seaborn')

# visualize data
fig, ax = plt.subplots(1,1)
sns.boxplot(data=tab, palette='tab10')
ax.set_xticklabels(tab.columns, rotation=90)
ax.set_title('Original Data')


###############################
# standardize data (zero mean, unit variance)
XX = tab.values
stand = StandardScaler()
XX_std = stand.fit_transform(XX)
tab_std = pd.DataFrame(XX_std, columns=tab.columns)

fig, ax = plt.subplots(1,2)
sns.boxplot(data=tab, ax=ax[0], palette='tab10')
ax[0].set_xticklabels(tab.columns, rotation=90)
ax[0].set_title('Original Data')
sns.boxplot(data=tab_std, ax=ax[1], palette='tab10')
ax[1].set_xticklabels(tab_std.columns, rotation=90)
ax[1].set_title('Standardized Data')


###############################
# perform PCA
pca_fun = PCA()
XX_std_pca = pca_fun.fit_transform(XX_std)

# visualize variance explained
fig, ax = plt.subplots(1,1)
ax.plot(np.cumsum(pca_fun.explained_variance_ratio_), ls='-', marker='o')
ax.axhline(0.95, color='red', ls='--')
ax.set_title('Cumulative Explained Variance Ratio')

# visualize projection on first 2 PCs
fig, ax = plt.subplots(1,1)
ax.scatter(XX_std_pca[:,0], XX_std_pca[:,1])
ax.set_title('Data projected on PC1 and PC2')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
























