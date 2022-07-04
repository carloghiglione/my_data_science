# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:56:54 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from scipy.stats import chi2, f, multivariate_normal
from tqdm import tqdm

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


##################################################################################
# Read Data
m1 = np.array([1.0, 1.0])
c1 = np.array([[2.0, 0.3], [0.3, 0.5]])
m2 = np.array([1.0, 1.0])
c2 = np.array([[0.5, 0.1], [0.1, 0.9]])
# c2 = np.array([[2.0, 0.3], [0.3, 0.5]])

X1 = multivariate_normal(m1, c1).rvs(100)
X2 = multivariate_normal(m2, c2).rvs(200)
n1 = X1.shape[0]
n2 = X2.shape[0]
d = X1.shape[1]
n = n1 + n2

fix, ax = plt.subplots(1,1)
ax.scatter(X1[:,0], X1[:,1])
ax.scatter(X2[:,0], X2[:,1])


###################################################################################
# Parametric Test (Hotellings-T Test, same covariance)
# H0: mu1 - mu2 = d0
# H1: mu1 - mu2 != d0
d0 = np.array([0.0, 0.0])

mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)
Sp = ( (n1-1)*S1 + (n2-1)*S2 )/(n1+n2-2)
Sp_inv = np.linalg.inv(Sp)

n_p = 1/(1/n1 + 1/n2)
T0 = n_p * ( (mu1 - mu2) - d0 ).T @ Sp_inv @ ( (mu1 - mu2) - d0 )

pval = 1 - f.cdf(x=T0*(n1+n2-1-d)/(d*(n1+n2-2)), dfn=n1+n2-1-d, dfd=d)

print(f'Hotellings T-test: pavlue={pval}')


###################################################################################
# Asymptotic Test (Welch-T Test, different covariance)
# H0: mu1 - mu2 = d0
# H1: mu1 - mu2 != d0
d0 = np.array([0.0, 0.0])

mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)
Sp = S1/n1 + S2/n2
Sp_inv = np.linalg.inv(Sp)

T0 = ( (mu1 - mu2) - d0 ).T @ Sp_inv @ ( (mu1 - mu2) - d0 )

pval = 1 - chi2(df=d).cdf(x = T0)

print(f'Welch T-test: pavlue={pval}')



# ###################################################################################
# Permutational Test
# H0: mu1 = mu2
# H1: mu1 != mu2

mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)
Xp = np.vstack((X1, X2))

def statistics(X1, X2):
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    return np.sum( (mu1 - mu2)**2 )

B = 10000
T0 = statistics(X1, X2)

T_list = []
for b in tqdm(range(B)):
    idx = np.arange(n)
    np.random.shuffle(idx)
    Xp_b = Xp[idx,:].copy()
    X1_b = Xp_b[:n1,:]
    X2_b = Xp_b[n1:,:]
    T_b = statistics(X1_b, X2_b)
    T_list.append(T_b)

pval = np.sum( np.array(T_list) >= T0 )/B
print(f'Permutational Test: pvalue={pval}')

fig, ax = plt.subplots(1,1)
sns.distplot(T_list, kde=True, ax=ax, label='T Distribution')
ax.axvline(T0, color='red', label='T0')
ax.legend()
ax.set_title('T Permutational Distribution')




