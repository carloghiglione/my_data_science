# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:15:12 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from scipy.stats import chi2, f
from scipy.stats import ttest_ind, mannwhitneyu, ranksums
from scipy.stats import shapiro, bartlett, levene
from tqdm import tqdm

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


##################################################################################
# Read Data
X1 = np.random.normal(2, 1, size=300)
X2 = np.random.normal(2, 1, size=200)

n1 = X1.shape[0]
n2 = X2.shape[0]
n = n1 + n2
d = 1


##################################################################################
# Test Variance
t_bart = bartlett(X1, X2)
t_lev = levene(X1, X2)      # can customize center (mean, median) and proprtion of trimming

print(f'Bartlett Test: pvalue={t_bart[1]}')
print(f'Levene Test: pvalue={t_lev[1]}')


###################################################################################
# Parametric (Hotellings-T, Equal Variance)
# H0: mu1 = mu2
# H1: mu1 != mu2
t_test = ttest_ind(X1, X2, equal_var=True)

print(f'T-Test: pvalue={t_test[1]}')


###################################################################################
# Parametric (Welch-t, Different Variance Asymptotic)
# H0: mu1 = mu2
# H1: mu1 != mu2
t_test = ttest_ind(X1, X2, equal_var=False)

print(f'T-Test: pvalue={t_test[1]}')


###################################################################################
# Nonparametric (Mann-Whitney-U)
# H0: law(X1) = law(X2)
# H1: law(X1) != law(X2) 
mwu_test = mannwhitneyu(X1, X2)

print(f'Mann-Whitney-U Test: pvalue={mwu_test[1]}')


###################################################################################
# Nonparametric (Wicoxon Rank-Sum)
# H0: law(X1) = law(X2)
# H1: P(X1 > X2) != 0.5  (values in one sample are more probable to be larger)
wrs_test = ranksums(X1, X2)

print(f'Wicoxon Rank-Sum Test: pvalue={wrs_test[1]}')


####################################################################################
# Permutational
# H0: mu1 = mu2
# H1: mu1 != mu2

mu1 = np.mean(X1)
mu2 = np.mean(X2)
Xp = np.hstack((X1, X2))

def statistics(X1, X2):
    mu1 = np.mean(X1)
    mu2 = np.mean(X2)
    return (mu1 - mu2)**2

B = 10000
T0 = statistics(X1, X2)

T_list = []
for b in tqdm(range(B)):
    idx = np.arange(n)
    np.random.shuffle(idx)
    Xp_b = Xp[idx].copy()
    X1_b = Xp_b[:n1]
    X2_b = Xp_b[n1:]
    T_b = statistics(X1_b, X2_b)
    T_list.append(T_b)

pval = np.sum( np.array(T_list) >= T0 )/B
print(f'Permutational Test: pvalue={pval}')

fig, ax = plt.subplots(1,1)
sns.distplot(T_list, kde=True, ax=ax, label='T Distribution')
ax.axvline(T0, color='red', label='T0')
ax.legend()
ax.set_title('T Permutational Distribution')





