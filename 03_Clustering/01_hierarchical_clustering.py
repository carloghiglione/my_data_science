# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:31:18 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent, cophenet
from sklearn.metrics import silhouette_score

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


####################################################
# read the data (generate them)
random_state = 1234 ## another interesting example can be generated using the seed 36
no_clusters = 3
no_samples = 150

X, y = make_blobs(centers=no_clusters, n_samples=no_samples, random_state=random_state)

fig, ax = plt.subplots(1,1)
ax.scatter(X[:,0], X[:,1])
fig.tight_layout()


####################################################
# Clustering (create the dendrogram)
# 'single',  'complete', 'average', 'ward', 'centroid', 'median'
method = 'complete'
Z = linkage(X, method=method, metric='euclidean')

fig, ax = plt.subplots(1,1)
dendrogram(Z)
ax.set_title(f'Dendrogram, method: {method}')
ax.set_xticklabels([])
fig.tight_layout()


####################################################
# Inconsistency Criterion (cut when it is high)
inconsistency = inconsistent(Z, d=10)

for i in reversed(range(1,15)):
    print("from %d to %d => Inconsistency %.3f"%(i,i+1,inconsistency[-i][3]))
    
 

#####################################################
# Knee-Elbow Analysis
wss_values = []
bss_values = []
silho_values = []
k_max = 20

for k in range(1,k_max):
    clustering = fcluster(Z, k, criterion='maxclust')
    centroids = [np.mean(X[clustering==c],axis=0) for c in range(1,k+1)]
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
        silho_values.append(silhouette_score(X, clustering))


fig = plt.figure()
plt.plot(np.arange(1,k_max), wss_values, ls='-', marker='o', label='WSS')
plt.plot(np.arange(1,k_max), bss_values, ls='-', marker='o', label='BSS')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('BSS & WSS')
plt.xticks(np.arange(1,k_max))
plt.legend()
plt.title('Hierarchical Clustering')


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



##################################################################################
# Display the Clustering
k_final = 3
final_clust = fcluster(Z, k_final, criterion='maxclust')

tab = pd.DataFrame(X, columns=['X1', 'X2'])

fig, ax = plt.subplots(1,1)
sns.scatterplot('X1', 'X2', data=tab, hue=final_clust, ax=ax, palette='tab10')
ax.set_title(f'Clusters: method={method}, k={k_final}')


###################################################################################
# Describe the clustering

fig, ax = plt.subplots(1,1, figsize=(8,4))
sns.countplot(final_clust, ax=ax)
ax.set_title('Clustering count')
plt.tight_layout()


final_clust_freq = np.bincount(final_clust)
idx = np.nonzero(final_clust_freq)[0]
final_clust_freq = final_clust_freq[idx]
print(f'Frequency of groups: {final_clust_freq}')


cdf = pd.DataFrame(columns = ['X1','X2'], data = X)
cdf['cluster'] = final_clust
print(cdf.groupby(by=['cluster']).describe())

coph_index, coph_dists= cophenet(Z, pdist(X))
print(f'Cophenetic index: {coph_index}')


# Create crosstab
ct = pd.crosstab(final_clust, y)
print(ct)
















