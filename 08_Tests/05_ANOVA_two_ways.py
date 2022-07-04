# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 22:16:33 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, shapiro, wilcoxon, levene, bartlett, f_oneway
from scipy.stats import t
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


#################################################################################
# Import Data
df_read = sm.datasets.get_rdataset("Moore", "carData", cache=True).data

group1 = df_read['fcategory'].values
group2 = df_read['partner.status'].values
X = df_read['fscore'].values

df = pd.DataFrame({'X': X})
df['group1'] = group1
df['group2'] = group2
df['group12'] = df['group1'] + df['group2']

n = df.shape[0]
g1 = df['group1'].nunique()
g2 = df['group2'].nunique()
g12 = df['group12'].nunique()
n1_list = df.groupby('group1').count()
n2_list = df.groupby('group2').count()
n12_list = df.groupby('group12').count()


fig, ax = plt.subplots(1,2)
sns.boxplot(x='group1', y='X', data=df, ax=ax[0])
sns.boxplot(x='group2', y='X', data=df, ax=ax[1])

fig, ax = plt.subplots(1,1)
sns.boxplot(x='group12', y='X', data=df, ax=ax)

fig, ax = plt.subplots(1,2)
interaction_plot(x=df['group1'], trace=df['group2'], response=df['X'], ax=ax[0])   # x can be continuous too
interaction_plot(x=df['group2'], trace=df['group1'], response=df['X'], ax=ax[1])


##################################################################################
# ANOVA with Interaction
mod_lm = ols('X ~ group1 + group2 + group1:group2', data=df).fit()
table = sm.stats.anova_lm(mod_lm, typ=2) # Type=2 to see it as in R

print('\ngroup1 + group2 + group1:group2')
print(table)
# should check hypotesis, see 04_ANOVA.py


##################################################################################
# ANOVA no Interaction
mod_lm = ols('X ~ group1 + group2', data=df).fit()
table = sm.stats.anova_lm(mod_lm, typ=2) # Type=2 to see it as in R

print('\ngroup1 + group2')
print(table)
# should check hypotesis, see 04_ANOVA.py


##################################################################################
# ANOVA one way
mod_lm = ols('X ~ group1', data=df).fit()
table = sm.stats.anova_lm(mod_lm, typ=2) # Type=2 to see it as in R

print('\ngroup1')
print(table)
# should check hypotesis, see 04_ANOVA.py

