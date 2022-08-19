# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:59:20 2022

@author: Utente
"""

from sklearn.datasets import make_blobs
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#####################################################
# read the data (generate them)
random_state = 1234 ## another interesting example can be generated using the seed 36
no_clusters = 3
no_samples = 150

X, true_group = make_blobs(centers=no_clusters, n_samples=no_samples, random_state=random_state)
tab = pd.DataFrame(X, columns=['X1', 'X2'])

##################
# Goal: perform k-means clustering 

#######################################################################################
#######################################################################################
# imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


fig, ax = plt.subplots(1,1)
sns.scatterplot('X1','X2', data=tab, ax=ax)
ax.set_title('Data')

################################
# k-means clustering
X = tab.values
k_final = 3
clust = KMeans(n_clusters=k_final).fit_predict(X)

tab['class'] = clust

fig, ax = plt.subplots(1,1)
sns.scatterplot('X1', 'X2', data=tab, hue=clust, palette='tab10')
ax.set_title(f'K-Means Clutsering, k = {k_final}')


##############################
# describe and evaluate clustering

# evalutate clustering, Silhouette Score
silho_score = silhouette_score(X, clust)

# describe clusters
tab.groupby('class').describe()

# visualize numerosity
num_clust = tab.groupby('class').size()
fig, ax = plt.subplots(1,1)
sns.barplot(x=num_clust.index, y=num_clust, ax=ax)
ax.set_title('Cluster Numerosity')
















