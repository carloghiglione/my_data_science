# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 08:29:21 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from scipy.stats import chi2, f
from tqdm import tqdm

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


##################################################################################
# Read Data
df = pd.read_csv('data/extra.txt', sep=' ')
X = df.values
X = np.random.randn(1000, 2)
n = X.shape[0]
d = X.shape[1]


##################################################################################
# Check Gaussianity
S = np.cov(X.T)
S_inv = np.linalg.inv(S)
mu = np.mean(X, axis=0)
dist2 = np.diag( (X - mu) @ S_inv @ (X - mu).T )

fig, ax = plt.subplots(1,1)
sns.distplot(dist2, ax=ax, kde=True, label='Normal Data')
xx = np.arange(0, 20, 0.01)
ax.plot(xx, chi2(df=d).pdf(xx), label=f'Chi2({d})')
ax.legend()
ax.set_title('Distance of Norml Data')

# MCD = EllipticEnvelope(contamination=0.01)
# MCD.fit(X)
# norm_out = MCD.predict(X)


##################################################################################
# Mean Test (Parametric)
# H0: mu = mu0
# H1: mu != mu0
from scipy.stats import ttest_1samp   # solo caso univariato 

# mu0 = np.array([120000, 52000])
mu0 = np.array([0.5, 0.5])
T0 = n * (mu - mu0).T @ S_inv @ (mu - mu0)
pval = 1 - f.ppf(q=T0*(n-d)/(d*(n-1)), dfn=n-d, dfd=d)

print(f'T-Test: pvalue={pval}')



###################################################################################
# Mean Test (Permutational)
def statistics(X, mu0):
    mu = np.mean(X, axis=0)
    return np.sum( (mu - mu0)**2 )

def permute(X, mu0):
    n = X.shape[0]
    d = X.shape[1]
    return mu0 + (X.copy() - mu0)*np.random.choice([-1,1], size=n, replace=True).reshape(-1,1)

# mu0 = np.array([120000, 52000])
mu0 = np.array([0.5, 0.5])
T0 = statistics(X, mu0)
B = 10000

T_list = []
for b in tqdm(range(B)):
    X_b = permute(X, mu0)
    T_b = statistics(X_b, mu0)
    T_list.append(T_b)

pval = np.sum(np.array(T_list) >= T0)/B
print(f'Permutation test: pvalue={pval}')

fig, ax = plt.subplots(1,1)
sns.distplot(T_list, ax=ax, kde=True, label='T Distribution')
ax.axvline(T0, color='red', label='T0')
ax.legend()
ax.set_title('T Permutational Distribution')

















