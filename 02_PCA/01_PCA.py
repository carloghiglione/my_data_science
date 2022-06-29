# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:12:50 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import datasets

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


####################################################################
# Read the data
df = pd.read_csv('data/tourists.txt', sep=' ')
tab = df.iloc[:,2:10]
tab_cols = tab.columns
n_cols = len(tab_cols)
df = df.rename(columns={'Month':'class'})


# dataset = datasets.load_iris()

# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# df.columns = ['X1', 'X2', 'X3', 'X4']
# df['class'] = dataset.target
# df['class'] = pd.Categorical(df['class'])    # set categorical variable

# tab = df.iloc[:,0:4]
# tab_cols = tab.columns
# n_cols = len(tab_cols)


std_scaler = StandardScaler()
tab_std = std_scaler.fit_transform(tab.to_numpy())
tab_std = pd.DataFrame(tab_std, columns=tab_cols)

fig, ax = plt.subplots(1,2)
tab.boxplot(ax = ax[0])
tab_std.boxplot(ax = ax[1])
ax[0].set_title('Original Data')
ax[1].set_title('Scaled Data')
fig.tight_layout()



######################################################################
# PCA on original data (sembra dare problemi)
pca = PCA()
tab_pca = pca.fit_transform(tab)
tab_pca = pd.DataFrame(tab_pca, columns=['PC'+str(a+1) for a in range(n_cols)])

fig, ax = plt.subplots(1,1)
ax.plot(np.cumsum(pca.explained_variance_ratio_), ls='-', marker='o')
ax.set_title('Cumulative Explained Variance Ratio')
fig.tight_layout()

tab_pca_loads = pca.components_

fig, ax = plt.subplots(1,2, sharey=True)
for a in range(2):
    sns.barplot(x=tab_cols, y=tab_pca_loads[a], ax=ax[a], color='royalblue')
    ax[a].set_xticklabels(tab_cols, rotation=90)
    ax[a].set_title(f'Loadings PC{a+1}')
fig.tight_layout()
    
fig, ax = plt.subplots(1,1)
sns.scatterplot('PC1', 'PC2', data=tab_pca, hue=df['class'], ax=ax)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Data Projected')
fig.tight_layout()



#########################################################################
# PCA on scaled data
pca_std = PCA()
tab_std_pca = pca_std.fit_transform(tab_std)
tab_std_pca = pd.DataFrame(tab_std_pca, columns=['PC'+str(a+1) for a in range(n_cols)])

fig, ax = plt.subplots(1,1)
ax.plot(np.cumsum(pca_std.explained_variance_ratio_), ls='-', marker='o')
ax.set_title('Cumulative Explained Variance Ratio')

tab_std_pca_loads = pca_std.components_

fig, ax = plt.subplots(1,2, sharey=True)
for a in range(2):
    sns.barplot(x=tab_cols, y=tab_std_pca_loads[a], color='royalblue', ax=ax[a])
    ax[a].set_xticklabels(tab_cols, rotation=90)
    ax[a].set_title(f'Loadings PC{a+1}')
fig.tight_layout()

fig, ax = plt.subplots(1,1)
sns.scatterplot('PC1', 'PC2', data=tab_std_pca, hue=df['class'], ax=ax)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Data Scaled Projected')
fig.tight_layout()



#########################################################################
# Project new datum on PCA space
# x_new = np.array([30, 23, 36, 22, 65, 19, 5, 15])
x_new = tab.iloc[1,:].values.reshape(-1, n_cols)
x_new_std = std_scaler.transform(x_new)

# non funziona
x_proj = tab_pca_loads @ x_new.T
print(x_proj)
print(tab_pca.iloc[1,:])

# funziona
x_proj_std = tab_std_pca_loads @ x_new_std.T
print(x_proj_std)
print(tab_std_pca.iloc[1,:])



###########################################################################
# Directly select the proportion of variance to keep (keeps one more)
var_propr = 0.975
pca_std_var = PCA(var_propr)
tab_std_pca_var = pca_std_var.fit_transform(tab_std)
tab_std_pca_var = pd.DataFrame(tab_std_pca_var, columns=['PC'+str(a+1) for a in range(tab_std_pca_var.shape[1])])


fig, ax = plt.subplots(1,1)
sns.scatterplot('PC1', 'PC2', data=tab_std_pca_var, hue=df['class'], ax=ax)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Data Scaled Projected')
fig.tight_layout()
























