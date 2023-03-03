# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:02:48 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn import mixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


####################################################
# read the data (generate them)
np.random.seed(1234)
n_clusters = 3
n_samples = 500

# X, y = make_blobs(centers=n_clusters, n_samples=n_samples, random_state=random_state)

C1 = np.array([[0.1, -0.2], [1.7, 0.4]])
C2 = np.array([[-1.0, 0.4], [-0.2, -0.1]])
C3 = 0.6*np.array([[1.0, 0.0], [0.0, 1.0]])
X1 = np.dot(np.random.randn(n_samples//n_clusters, 2), C1) + np.array([-2, 0.25])  # general
X2 = np.dot(np.random.randn(n_samples//n_clusters, 2), C2)  # general
X3 = np.dot(np.random.randn(n_samples//n_clusters, 2), C3) + np.array([-4.5,0])  # spherical

X = np.concatenate([X1, X2, X3])

fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.scatter(X[:,0], X[:,1])
# ax.set_aspect('equal', adjustable='box')
plt.axis('equal')
fig.tight_layout()



#####################################################
# Gaussian Mixture
n_clust = 3
gmm = mixture.GaussianMixture(n_components=n_clust, covariance_type='full')  # 'full', 'tied', 'diag', 'spherical'
clust = gmm.fit_predict(X)

tab = pd.DataFrame(X, columns=['X1', 'X2'])
tab['clust'] = clust

fig, ax = plt.subplots(1,1)
sns.scatterplot(x='X1', y='X2', data=tab, hue='clust', ax=ax, palette='tab10')
ax.set_title(f'GMM: k={n_clust}')


silhouette_avg = silhouette_score(X, clust)



#######################################################
# Model Selection
bic_values = []
aic_values = []
silho_values = []
k_max = 20

for k in range(2,k_max):
    curr_gmm = mixture.GaussianMixture(n_components=k, covariance_type='full') # 'full', 'tied', 'diag', 'spherical'
    clustering = curr_gmm.fit_predict(X)
    
    silho_values.append(silhouette_score(X, clustering))
    bic_values.append(curr_gmm.bic(X))
    aic_values.append(curr_gmm.aic(X))


fig, ax = plt.subplots(1,1)
ax.plot(np.arange(2,k_max), silho_values, ls='-', marker='o')
ax.set_xlabel('Number of clusters')
ax.set_xticks(np.arange(2,k_max))
ax.set_title('GMM, Silhouette Score')

fig, ax = plt.subplots(1,1)
ax.plot(np.arange(2,k_max), bic_values, ls='-', marker='o')
ax.set_xlabel('Number of clusters')
ax.set_xticks(np.arange(2,k_max))
ax.set_title('GMM, BIC Score')

fig, ax = plt.subplots(1,1)
ax.plot(np.arange(2,k_max), aic_values, ls='-', marker='o')
ax.set_xlabel('Number of clusters')
ax.set_xticks(np.arange(2,k_max))
ax.set_title('GMM, AIC Score')



#################################################################################
# Fitted density
from matplotlib.colors import LogNorm

# display predicted scores by the model as a contour plot
xx1 = np.linspace(-10.0, 4.0)
xx2 = np.linspace(-2.0, 2.0)
XX1, XX2 = np.meshgrid(xx1, xx2)
XX = np.array([XX1.ravel(), XX2.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(XX1.shape)

fig, ax = plt.subplots(1,1)
CS = plt.contour(XX1, XX2, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 25), cmap='viridis')
CB = plt.colorbar(CS, shrink=0.8, extend="both", ax=ax)
ax.scatter(X[:, 0], X[:, 1], s=10)

plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")



