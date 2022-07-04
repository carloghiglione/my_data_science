# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:12:06 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, shapiro, wilcoxon
from tqdm import tqdm

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


##################################################################################
# Read Data
df = pd.read_csv('data/parziali.txt', sep=' ')
X = df.values
n = X.shape[0]
d = X.shape[1]


fig, ax = plt.subplots(1,1)
ax.plot(X.T)
ax.plot(np.mean(X, axis=0), ls='--', color='black', lw=4)
plt.xticks(ticks=np.arange(d), labels=df.columns[:2])


###################################################################################
# Gaussianity Check of Components
shap_pvals = []
for a in range(d):
    shap_pvals.append(shapiro(X[:,a])[1])
print(f'Shapiro Test: {shap_pvals}')


###################################################################################
# Parametric Test (Hotellings T)
# H0: mean(exp0) = mean(exp1)
# H1: mean(exp0) != mean(exp1)
exp0 = X[:,0]
exp1 = X[:,1]

t_test = ttest_rel(exp0, exp1)

print(f'Paired T-Test: pvalue={t_test[1]}')

# alternative: set diff = exp0 - exp1, test one population mean = 0


####################################################################################
# Nonparametric Test (Wilcoxon)
# H0: law(exp0) = law(exp1)
# H1: law(exp0) != law(exp1)
exp0 = X[:,0]
exp1 = X[:,1]

w_test = wilcoxon(exp0, exp1)

print(f'Paired Wilcoxon-Test: pvalue={w_test[1]}')

# alternative: set diff = exp0 - exp1, test one population mean = 0


####################################################################################
# Permutational Test
# H0: mean(exp0) - mean(exp1) = d0
# H1: mean(exp0) - mean(exp1) != d0
diff = X[:,0] - X[:,1]
d0 = 0

def statistics(X, d0):
    mu = np.mean(X)       # also median is a possibility
    return (mu - d0)**2

def permute(X, d0):
    n = X.shape[0]
    return d0 + (X.copy() - d0)*np.random.choice([-1,1], size=n, replace=True).reshape(-1,)


T0 = statistics(diff, d0)
B = 10000

T_list = []
for b in tqdm(range(B)):
    diff_b = permute(diff, d0)
    T_b = statistics(diff_b, d0)
    T_list.append(T_b)

pval = np.sum(np.array(T_list) >= T0)/B
print(f'Permutation test: pvalue={pval}')

fig, ax = plt.subplots(1,1)
sns.distplot(T_list, ax=ax, kde=True, label='T Distribution')
ax.axvline(T0, color='red', label='T0')
ax.legend()
ax.set_title('T Permutational Distribution')









