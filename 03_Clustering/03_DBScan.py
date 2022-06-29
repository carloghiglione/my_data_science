# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:18:27 2022

@author: Utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import seaborn as sns

#plt.style.use('seaborn')


############################################################################
# Read the data
tab = pd.read_csv('data/Chamaleon.txt', sep=' ')

fig, ax = plt.subplots(1,1)
sns.scatterplot('x', 'y', data=tab, ax=ax)
ax.set_title('Data')
fig.tight_layout()


############################################################################
# DBScan
eps = 10               # maximum distance between two samples for one to be considered as in the neighborhood of the other
min_samples = 10       # number of samples in a neighborhood for a point to be considered as a core point

dbscan_clust = DBSCAN(eps=eps, min_samples=min_samples).fit(tab.values)
clust = dbscan_clust.labels_


fig, ax = plt.subplots(1,1, figsize=(8,4))
sns.countplot(clust, ax=ax)
ax.set_title('Clustering count')
plt.tight_layout()


n_clusters_ = len(set(clust)) - (1 if -1 in clust else 0)
noisy_points = clust == -1
cluster_points = ~noisy_points


clust_freq = np.bincount(clust[cluster_points])
idx = np.nonzero(clust_freq)[0]
clust_freq = clust_freq[idx]
print(f'Frequency of groups: {clust_freq}')
print("Number of clusters = %d"%n_clusters_)
print("Number of cluster points = %d"%sum(cluster_points))
print("Number of noisy points = %d"%sum(noisy_points))


tab['clust'] = clust

fig, ax = plt.subplots(1,1)
sns.scatterplot('x', 'y', data=tab, hue='clust', ax=ax, palette='tab10')
ax.set_title('Data DBScan Clustering')
fig.tight_layout()


fig, ax = plt.subplots(1,1)
ax.scatter(tab['x'], tab['y'], c=clust, cmap='tab20')
ax.set_title('Data DBScan Clustering')
fig.tight_layout()








