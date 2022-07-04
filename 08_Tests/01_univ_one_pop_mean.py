# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:52:36 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from scipy.stats import chi2, f
from scipy.stats import shapiro
from tqdm import tqdm

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


##################################################################################
# Read Data
X = np.random.normal(2, 4, size=300)
n = X.shape[0]
d = 1

##################################################################################
# Check Gaussianity
test_shap = shapiro(X)

print(f'Shapiro Test: pvalue={test_shap[1]}')


##################################################################################
# Mean Test (Parametric)
# H0: mu = mu0
# H1: mu != mu0
from scipy.stats import ttest_1samp   # solo caso univariato 

mu = np.mean(X)
mu0 = 2.5
t_test = ttest_1samp(X, mu0, alternative='two-sided')

print(f'T-Test: pvalue={t_test[1]}')



###################################################################################
# Mean Test (Permutational)
def statistics(X, mu0):
    mu = np.mean(X)       # also median is a possibility
    return (mu - mu0)**2

def permute(X, mu0):
    n = X.shape[0]
    return mu0 + (X.copy() - mu0)*np.random.choice([-1,1], size=n, replace=True).reshape(-1,)

# mu0 = np.array([120000, 52000])
mu0 = 2.5
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









