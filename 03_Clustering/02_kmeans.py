# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:52:53 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
import pandas as pd
import seaborn as sns

plt.style.use('seaborn')

###################################################################
# read data (generate)
random_state = 1234 ## another interesting example can be generated using the seed 36
no_clusters = 3
no_samples = 150

X, y = make_blobs(centers=no_clusters, n_samples=no_samples, random_state=random_state)

fig, ax = plt.subplots(1,1)
ax.scatter(X[:,0], X[:,1])
fig.tight_layout()


####################################################################
# KMeans clustering
n_clus = 3
clust = KMeans(n_clusters=n_clus).fit_predict(X)

tab = pd.DataFrame(X, columns=['X1', 'X2'])
tab['clust'] = clust

fig, ax = plt.subplots(1,1)
sns.scatterplot('X1', 'X2', data=tab, hue='clust', ax=ax, palette='tab10')
ax.set_title(f'KMeans: k={n_clus}')


silhouette_avg = silhouette_score(X, clust)

#####################################################################
# Knee-Elbow Analysis and Silhouette
wss_values = []
bss_values = []
silho_values = []
k_max = 20

for k in range(1,k_max):
    clustering = KMeans(n_clusters=k).fit(X)
    centroids = clustering.cluster_centers_ 
    cdist(X, centroids, 'euclidean')
    D = cdist(X, centroids, 'euclidean')
    cIdx = np.argmin(D,axis=1)
    d = np.min(D,axis=1)

    avgWithinSS = sum(d)/len(X)

    # Total with-in sum of square
    wss = sum(d**2)

    tss = sum(pdist(X)**2)/len(X)
    
    bss = tss-wss
    
    wss_values += [wss]
    bss_values += [bss]
    
    if k > 1:
        silho_values.append(silhouette_score(X, clustering.labels_))
    
    
fig = plt.figure()
plt.plot(np.arange(1,k_max), wss_values, ls='-', marker='o', label='WSS')
plt.plot(np.arange(1,k_max), bss_values, ls='-', marker='o', label='BSS')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('BSS & WSS')
plt.xticks(np.arange(1,k_max))
plt.legend()
plt.title('KMeans Clustering')


fig = plt.figure()
plt.plot(np.arange(1,k_max), np.array(wss_values)/(np.array(wss_values) + np.array(bss_values)), ls='-', marker='o')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,k_max))
plt.legend()
plt.title('KMeans Clustering, Within vs Tot')


fig = plt.figure()
plt.plot(np.arange(2,k_max), silho_values, ls='-', marker='o')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.xticks(np.arange(2,k_max))
plt.legend()
plt.title('KMeans Clustering, Within vs Tot')













