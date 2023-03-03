# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:47:19 2022

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
# Goal: perform hierarchical clustering 


################################################################################################
################################################################################################
# imports
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


#####################################################
# visualize data
fig, ax = plt.subplots(1,1)
sns.scatterplot(data=tab, x='X1', y='X2', ax=ax)
ax.set_title('Data')

####################################################
# hierarchical clustering

method = 'complete'  # single, complete, average, ward
metric = 'euclidean'

Z = linkage(X, method=method, metric=metric)

# visualize dendrogram
fig, ax = plt.subplots(1,1)
dendrogram(Z, ax=ax)
ax.set_title(f'Hierachical clustering, method={method}')

# set number of clusters
k_final = 3
final_clust = fcluster(Z, k_final, criterion='maxclust')

fig, ax = plt.subplots(1,1)
sns.scatterplot(data=tab, x='X1', y='X2', hue=final_clust, ax=ax, palette='tab10')
ax.set_title(f'Hierarchical Clustering: method={method}, k={k_final}')


#######################################################
# describe and evaluate clusters
tab['class'] = final_clust

clust_descr = tab.groupby('class').describe()

# visualize numerosity
clust_num = tab.groupby('class').size()
fig, ax = plt.subplots(1,1)
sns.barplot(x=clust_num.index, y=clust_num, ax=ax)
ax.set_title('Cluster Numerosity')

# evalutate clustering, Silhouette Score
silho_score = silhouette_score(X, final_clust)



















